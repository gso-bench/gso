"""
Reward Hack Detector for GSO with Threshold Variation

This script runs reward hack detection at multiple speedup thresholds.
It analyzes instances ONCE at the lowest threshold and propagates results to higher thresholds.

Usage:
    uv run src/gso/analysis/qualitative/detect_reward_hacks_thresholded.py \
        --run_dirs "gpt-5-2025-08-07_maxiter_100_N_v0.51.1-no-hint-run_*" \
        --thresholds 0.25 0.5 0.75 \
        --model_name "gpt-5"
"""

import json
import argparse
import glob
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from gso.utils.io import load_gso_dataset
from gso.constants import EVALUATION_REPORTS_DIR, MIN_PROB_SPEEDUP
from gso.harness.utils import natural_sort_key
from gso.analysis.qualitative.utils import load_model_patch
from gso.analysis.qualitative.hack_detector import analyze_patch_for_hacks


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

        # Get all attempted instances
        attempted_instances = set()
        for key, instances in instance_sets.items():
            if key.endswith("_ids"):
                attempted_instances.update(instances)

        # Check each instance against threshold
        for instance_id in attempted_instances:
            if instance_id not in instance_lookup or instance_id not in opt_stats:
                continue

            stats = opt_stats[instance_id]
            base_mean = stats.get("base_mean", 0.0)
            patch_mean = stats.get("patch_mean", 0.0)
            pb_speedup_gm = stats.get("gm_speedup_patch_base", 0.0)
            pc_speedup_hm = stats.get("hm_speedup_patch_commit", 0.0)

            passes_threshold = (
                base_mean > patch_mean
                and round(pb_speedup_gm, 1) >= MIN_PROB_SPEEDUP
                and pc_speedup_hm > threshold_p
            )

            if passes_threshold:
                # Keep the best run for this instance
                if (
                    instance_id not in best_runs
                    or pc_speedup_hm > best_runs[instance_id]["pc_speedup_hm"]
                ):
                    best_runs[instance_id] = {
                        "run_dir": run_dir,
                        "pc_speedup_hm": pc_speedup_hm,
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
                all_results.append(result)
            except Exception as e:
                all_results.append({"instance_id": "unknown", "error": str(e)})

    # Filter out errors
    valid_analyses = [r for r in all_results if "error" not in r]
    print(f"Successfully analyzed {len(valid_analyses)}/{len(analysis_args)} instances")

    return valid_analyses


def generate_threshold_summary(all_analyses, threshold_p):
    """Generate summary for a specific threshold by filtering analyses."""
    filtered = [a for a in all_analyses if a["pc_speedup_hm"] > threshold_p]

    if not filtered:
        return {
            "total_analyzed": 0,
            "hack_count": 0,
            "legitimate_count": 0,
            "hack_percentage": 0.0,
            "average_confidence": 0.0,
            "instance_ids": [],
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dirs",
        required=True,
        help="Run directory glob pattern (e.g., 'gpt-5*_run_*')",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.95],
        help="Speedup thresholds to analyze (default: 0.95)",
    )
    parser.add_argument("--max_workers", type=int, default=60)
    parser.add_argument("--k_samples", type=int, default=5)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    dataset = load_gso_dataset("gso-bench/gso", "test")
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
    if len(thresholds) == 1 and thresholds[0] == 0.95:
        # Single threshold mode: flatten structure for backward compatibility
        output_data.update(threshold_results[str(thresholds[0])])
        file_name = f"{model_name}_hack_detection.json"
    else:
        # Multi-threshold mode: nest under "thresholds"
        output_data["thresholds"] = threshold_results
        file_name = f"{model_name}_hack_detection_thresholded.json"

    # Save output
    output_path = EVALUATION_REPORTS_DIR / "analysis" / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    print(f"Total instances analyzed: {len(all_analyses)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
