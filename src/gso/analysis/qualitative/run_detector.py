"""Run reward hack detection on final predictions or run directories.

Example:
    uv run python -m gso.analysis.qualitative.run_detector \
        --predictions_file output.gso.jsonl --report_path model.report.json \
        --model_name model --output_path detector.json
"""

import json
import argparse
import glob
import hashlib
import sys
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from gso.utils.io import load_gso_dataset
from gso.constants import EVALUATION_REPORTS_DIR, MIN_PROB_SPEEDUP
from gso.harness.utils import natural_sort_key
from gso.analysis.qualitative.utils import load_model_patch
from gso.analysis.qualitative.hack_detector import (
    analyze_patch_for_hacks,
    JUDGE_MODEL,
    JUDGE_BASE_URL,
)


def get_instances_passing_threshold(report_paths, threshold_p, dataset):
    """
    Find instances that pass the threshold and identify the best run for each.

    Returns:
        dict: {instance_id: {"run_dir": str, "pc_speedup_hm": float}}
    """
    instance_lookup = {inst.instance_id: inst for inst in dataset}
    best_runs = {}

    for report_path in report_paths:
        with open(report_path, "r") as f:
            report = json.load(f)

        run_dir = Path(report_path).parent.name
        opt_stats = report.get("opt_stats", {})
        instance_sets = report.get("instance_sets", {})
        passed_ids = set(instance_sets.get("passed_ids", []))
        empty_patch_ids = set(instance_sets.get("empty_patch_ids", []))

        # Get all attempted instances
        attempted_instances = set()
        for key, instances in instance_sets.items():
            if key.endswith("_ids"):
                attempted_instances.update(instances)

        # Check each instance against threshold
        for instance_id in attempted_instances:

            # skip instances that don't have opt stats only in threshold mode
            if threshold_p > 0.0:
                if instance_id not in instance_lookup or instance_id not in opt_stats:
                    continue
            else:
                # skip empty patches in no-threshold mode
                if instance_id in empty_patch_ids:
                    continue

            stats = opt_stats.get(instance_id, {})
            base_mean = stats.get("base_mean", 0.0)
            patch_mean = stats.get("patch_mean", 0.0)
            pb_speedup_gm = stats.get("gm_speedup_patch_base", 0.0)
            pc_speedup_hm = stats.get("hm_speedup_patch_commit", 0.0)

            passes_threshold = (
                base_mean > patch_mean
                and round(pb_speedup_gm, 1) >= MIN_PROB_SPEEDUP
                and pc_speedup_hm > threshold_p
            )

            if passes_threshold or threshold_p == 0.0:
                # Keep the best run for this instance
                if (
                    instance_id not in best_runs
                    or pc_speedup_hm > best_runs[instance_id]["pc_speedup_hm"]
                ):
                    best_runs[instance_id] = {
                        "run_dir": run_dir,
                        "pc_speedup_hm": pc_speedup_hm,
                        "is_correct": instance_id in passed_ids,
                    }

    return best_runs


def analyze_single_instance(args):
    """Analyze a single instance - designed for parallel execution."""
    instance_id, instance, run_dir, k_samples = args

    model_patch = load_model_patch(instance_id, run_dir)
    if not model_patch:
        return {"instance_id": instance_id, "error": "No model patch found"}

    analysis = analyze_patch_for_hacks(
        instance_id,
        instance.repo,
        instance.api,
        instance.tests,
        instance.gt_diff,
        model_patch,
        k_samples=k_samples,
    )

    if "error" in analysis:
        return {"instance_id": instance_id, "error": analysis.get("error")}

    # Add metadata, but remove heavy data to save memory
    analysis["instance_id"] = instance_id
    analysis["repo"] = instance.repo
    analysis["api"] = instance.api
    analysis["run_dir"] = run_dir

    return analysis


def analyze_at_lowest_threshold(
    run_dirs_pattern, min_threshold, dataset, k_samples, max_workers
):
    """
    Analyze all instances passing the lowest threshold.
    Returns all analyses with pc_speedup_hm for filtering at higher thresholds.
    """
    # Find all run-level report files
    run_dirs = sorted(glob.glob(run_dirs_pattern), key=natural_sort_key)
    if not run_dirs:
        print(f"ERROR: No run directories found for pattern: {run_dirs_pattern}")
        return []

    pass_reports = []
    for run_dir in run_dirs:
        pass_files = glob.glob(f"{run_dir}/*.pass.report.json")
        pass_reports.extend(pass_files)

    if not pass_reports:
        print("ERROR: No pass reports found")
        return []

    print(f"Found {len(pass_reports)} pass reports across {len(run_dirs)} runs")

    # Get instances passing the lowest threshold
    best_runs = get_instances_passing_threshold(pass_reports, min_threshold, dataset)

    if not best_runs:
        print(f"No instances pass lowest threshold p={min_threshold}")
        return []

    print(f"{len(best_runs)} instances pass lowest threshold p={min_threshold}")

    # Prepare analysis arguments
    instance_lookup = {inst.instance_id: inst for inst in dataset}
    analysis_args = [
        (instance_id, instance_lookup[instance_id], run_info["run_dir"], k_samples)
        for instance_id, run_info in best_runs.items()
        if instance_id in instance_lookup
    ]

    # Run hack detection in parallel
    print(f"Analyzing {len(analysis_args)} instances ...")
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(analyze_single_instance, arg) for arg in analysis_args
        ]
        for future in tqdm(futures, desc="Analyzing instances", unit="patch"):
            try:
                result = future.result()
                if "error" not in result:
                    instance_id = result["instance_id"]
                    result["pc_speedup_hm"] = best_runs[instance_id]["pc_speedup_hm"]
                    result["is_correct"] = best_runs[instance_id]["is_correct"]
                all_results.append(result)
            except Exception as e:
                all_results.append({"instance_id": "unknown", "error": str(e)})

    # Filter out errors
    valid_analyses = [r for r in all_results if "error" not in r]
    print(f"Successfully analyzed {len(valid_analyses)}/{len(analysis_args)} instances")

    return valid_analyses


def generate_threshold_summary(all_analyses, threshold_p):
    """Generate summary for a specific threshold by filtering analyses."""
    filtered = [
        a
        for a in all_analyses
        if "error" not in a and (threshold_p == 0.0 or a["pc_speedup_hm"] > threshold_p)
    ]

    if not filtered:
        return {
            "total_analyzed": 0,
            "hack_count": 0,
            "legitimate_count": 0,
            "hack_percentage": 0.0,
            "average_confidence": 0.0,
            "instance_ids": [],
            "hacked_instances": [],
        }

    hack_count = sum(1 for a in filtered if a.get("is_reward_hack", False))
    confidences = [a.get("confidence", 0.0) for a in filtered]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total_analyzed": len(filtered),
        "hack_count": hack_count,
        "legitimate_count": len(filtered) - hack_count,
        "hack_percentage": (hack_count / len(filtered) * 100) if filtered else 0.0,
        "average_confidence": avg_confidence,
        "instance_ids": [a["instance_id"] for a in filtered],
        "hacked_instances": [
            a["instance_id"] for a in filtered if a.get("is_reward_hack", False)
        ],
    }


def print_score_summary(report_path: str, hacked_instances: list[str]):
    """Print score and score_hack_control from the report."""
    with open(report_path, "r") as f:
        report = json.load(f)

    total = report["summary"]["total_instances"]
    opt_commit = report["summary"]["opt_commit"]
    score = report["summary"].get("score", round(opt_commit / total * 100, 2))

    opt_commit_ids = set(report["instance_sets"].get("opt_commit_ids", []))
    hacked_in_opt_commit = opt_commit_ids & set(hacked_instances)
    legitimate_count = opt_commit - len(hacked_in_opt_commit)
    score_hack_control = round((legitimate_count / total) * 100, 2)

    print(f"\n===== Hack Detection Summary =====")
    print(f"Total instances: {total}")
    print(f"Hacks: {len(hacked_instances)}")
    print(f"Improved over commit: {opt_commit}")
    print(f"Hacks & improved over commit: {len(hacked_in_opt_commit)}")
    print(f"Legitimate: {legitimate_count}")
    print("=" * 10)
    print(f"score: {score}%")
    print(f"score_hack_control: {score_hack_control}%")
    print("=" * 10)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selected_predictions(predictions_file, report):
    rows = [
        json.loads(line)
        for line in Path(predictions_file).read_text().splitlines()
        if line.strip()
    ]
    selected = {row["instance_id"]: row for row in rows}
    if len(selected) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    sets = report["instance_sets"]
    attempted = set().union(
        *(
            set(ids)
            for key, ids in sets.items()
            if key.endswith("_ids") and key != "incomplete_ids"
        )
    )
    if (
        sets.get("incomplete_ids")
        or set(selected) != attempted
        or len(selected) != report["summary"]["total_instances"]
    ):
        raise ValueError("Predictions must exactly match the final report")
    empty = {key for key, row in selected.items() if not row["model_patch"].strip()}
    if empty != set(sets.get("empty_patch_ids", [])):
        raise ValueError("Empty patches differ between predictions and report")
    return {key: row for key, row in selected.items() if key not in empty}, sorted(
        empty
    )


def analyze_selected_predictions(args, dataset):
    """Audit explicitly selected attempts, with resumable, complete vote coverage."""
    report = json.loads(Path(args.report_path).read_text())
    selected, empty = load_selected_predictions(args.predictions_file, report)
    lookup = {inst.instance_id: inst for inst in dataset}
    if not set(selected) <= set(lookup):
        raise ValueError("Prediction IDs are absent from the dataset")
    output_path = Path(
        args.output_path
        or (
            EVALUATION_REPORTS_DIR
            / "analysis"
            / "hack_detection"
            / "no_threshold"
            / f"{args.model_name}_hack_detection_no_threshold.json"
        )
    )
    config = {
        "judge_model": JUDGE_MODEL,
        "base_url": JUDGE_BASE_URL,
        "reasoning_effort": "xhigh",
        "k_samples": args.k_samples,
        "predictions_sha256": sha256_file(args.predictions_file),
        "report_sha256": sha256_file(args.report_path),
        "detector_sha256": sha256_file(Path(__file__).with_name("hack_detector.py")),
        "runner_sha256": sha256_file(__file__),
        "utils_sha256": sha256_file(Path(__file__).with_name("utils.py")),
    }
    saved = {}
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if previous.get("config") != config:
            raise ValueError(
                "Existing checkpoint has different inputs or judge configuration"
            )
        saved = {r["instance_id"]: r for r in previous["all_analyses"]}
        if not set(saved) <= set(selected):
            raise ValueError("Checkpoint contains unselected instances")

    def save():
        analyses = [saved[key] for key in sorted(saved)]
        good = [r for r in analyses if "error" not in r]
        summary = generate_threshold_summary(good, 0.0)
        flagged_winners = set(summary["hacked_instances"]) & set(
            report["instance_sets"]["opt_commit_ids"]
        )
        total = report["summary"]["total_instances"]
        wins = report["summary"]["opt_commit"]
        payload = {
            "score": round(wins / total * 100, 2),
            "score_hack_control": (
                round((wins - len(flagged_winners)) / total * 100, 2)
                if len(good) == len(selected)
                else None
            ),
            "flagged_target_ids": sorted(flagged_winners),
            "model_name": args.model_name,
            "config": config,
            "predictions_file": str(args.predictions_file),
            "report_path": str(args.report_path),
            "k_samples": args.k_samples,
            "expected_count": len(selected),
            "skipped_empty_ids": empty,
            "pending_ids": sorted(set(selected) - set(saved)),
            "error_ids": sorted(r["instance_id"] for r in analyses if "error" in r),
            "complete": len(good) == len(selected),
            "all_analyses": analyses,
            **summary,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(output_path)

    def analyze(instance_id):
        inst = lookup[instance_id]
        row = selected[instance_id]
        old = saved.get(instance_id, {})
        result = analyze_patch_for_hacks(
            instance_id,
            inst.repo,
            inst.api,
            inst.tests,
            inst.gt_diff,
            row["model_patch"],
            k_samples=args.k_samples,
            initial_responses=old.get("individual_responses", []),
        )
        result.update(
            {
                "instance_id": instance_id,
                "repo": inst.repo,
                "api": inst.api,
                "model_name_or_path": row["model_name_or_path"],
                "patch_sha256": hashlib.sha256(row["model_patch"].encode()).hexdigest(),
                "is_correct": instance_id in report["instance_sets"]["passed_ids"],
                "meets_target": instance_id
                in report["instance_sets"]["opt_commit_ids"],
                "pc_speedup_hm": report.get("opt_stats", {})
                .get(instance_id, {})
                .get("hm_speedup_patch_commit", 0.0),
            }
        )
        return result

    pending = [
        key for key in sorted(selected) if key not in saved or "error" in saved[key]
    ]
    print(
        f"{args.model_name}: {len(selected)} patches, {len(empty)} empty, "
        f"{len(selected) - len(pending)} already complete",
        flush=True,
    )
    print(
        f"Judge {JUDGE_MODEL}, {args.k_samples} votes, "
        f"at most {args.max_workers * args.k_samples} concurrent requests",
        flush=True,
    )
    save()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(analyze, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                saved[key] = future.result()
            except Exception as exc:
                saved[key] = {
                    **saved.get(key, {}),
                    "instance_id": key,
                    "error": type(exc).__name__,
                }
            save()
            result = saved[key]
            verdict = result.get(
                "error", "flagged" if result.get("is_reward_hack") else "unflagged"
            )
            done = sum("error" not in r for r in saved.values())
            print(
                f"[{done}/{len(selected)}] {key}: {verdict} "
                f"({result.get('total_votes', 0)}/{args.k_samples} votes)",
                flush=True,
            )
    print(f"Results saved to: {output_path}", flush=True)
    good = [r for r in saved.values() if "error" not in r]
    if len(good) != len(selected):
        print(
            "INCOMPLETE: score adjustment withheld until all patches have full vote coverage."
        )
        return 1
    print_score_summary(
        args.report_path, [r["instance_id"] for r in good if r["is_reward_hack"]]
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run_dirs", help="Legacy run directory glob pattern")
    inputs.add_argument("--predictions_file", help="Final selected output.gso.jsonl")
    parser.add_argument("--output_path", help="Resumable detector report path")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--k_samples", type=int, default=10)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--report_path", type=str, required=True, help="Path to merged report"
    )
    parser.add_argument(
        "--reformat_reports", action="store_true", help="Reuse existing results"
    )
    args = parser.parse_args()
    if args.k_samples < 1 or args.max_workers < 1:
        parser.error("k_samples and max_workers must be positive")
    if args.predictions_file and args.thresholds != [0.0]:
        parser.error(
            "Selected prediction mode audits all nonempty patches; use thresholds 0.0"
        )

    # Reformat mode: reuse existing detection results
    if args.reformat_reports:
        detection_dir = (
            EVALUATION_REPORTS_DIR / "analysis" / "hack_detection" / "no_threshold"
        )
        detection_file = (
            detection_dir / f"{args.model_name}_hack_detection_no_threshold.json"
        )

        if not detection_file.exists():
            print(f"Error: No detection file found at {detection_file}")
            print("Run detector first without --reformat_reports")
            return

        with open(detection_file, "r") as f:
            detection_data = json.load(f)

        hacked_instances = detection_data.get("hacked_instances", [])
        print(f"Loaded {len(hacked_instances)} hacked instances from existing results")
        print_score_summary(args.report_path, hacked_instances)
        return

    # Load dataset
    dataset = load_gso_dataset("gso-bench/gso", "test")
    if args.predictions_file:
        return analyze_selected_predictions(args, dataset)
    model_name = args.model_name
    thresholds = sorted(args.thresholds)
    min_threshold = min(thresholds)

    print(f"\n{'='*80}")
    print(f"Running reward hack detection for: {model_name}")
    print(f"Thresholds: {thresholds}")
    print(f"{'='*80}\n")

    # Analyze once at the lowest threshold
    all_analyses = analyze_at_lowest_threshold(
        args.run_dirs,
        min_threshold,
        dataset,
        args.k_samples,
        args.max_workers,
    )

    if not all_analyses:
        print("No instances to analyze. Exiting.")
        return

    # Generate summaries for each threshold
    print(f"\nGenerating summaries for each threshold...")
    threshold_results = {}
    for threshold in thresholds:
        summary = generate_threshold_summary(all_analyses, threshold)
        threshold_results[str(threshold)] = summary
        print(
            f"  p={threshold:4.2f}: {summary['hack_count']:3d}/{summary['total_analyzed']:3d} hacks "
            f"({summary['hack_percentage']:5.1f}%), conf={summary['average_confidence']:.2f}"
        )

    # Build output structure
    output_data = {
        "model_name": model_name,
        "run_dirs_pattern": args.run_dirs,
        "k_samples": args.k_samples,
        "all_analyses": all_analyses,
    }

    # Choose output format based on thresholds
    analysis_dir = EVALUATION_REPORTS_DIR / "analysis" / "hack_detection"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if len(thresholds) == 1 and thresholds[0] == 0.95:
        # Single threshold mode
        output_data.update(threshold_results[str(thresholds[0])])
        file_name = f"{model_name}_hack_detection.json"
        output_path = analysis_dir / "default_threshold" / file_name
    elif min_threshold == 0.0:
        # No-threshold mode
        output_data.update(threshold_results[str(min_threshold)])
        file_name = f"{model_name}_hack_detection_no_threshold.json"
        output_path = analysis_dir / "no_threshold" / file_name
    else:
        # Multi-threshold mode
        output_data["thresholds"] = threshold_results
        file_name = f"{model_name}_hack_detection_thresholded.json"
        output_path = analysis_dir / "thresholded" / file_name

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Get hacked instances for score calculation
    if min_threshold == 0.0:
        hacked_instances = output_data.get("hacked_instances", [])
    else:
        threshold_key = "0.95" if "0.95" in threshold_results else str(max(thresholds))
        hacked_instances = threshold_results[threshold_key].get("hacked_instances", [])

    print_score_summary(args.report_path, hacked_instances)


if __name__ == "__main__":
    sys.exit(main())
