"""
Reward Hack Detector for GSO with Threshold Variation

This script runs reward hack detection at multiple speedup thresholds,
analyzing the best run (highest pc_speedup_hm) for each instance at each threshold.

Usage:
    uv run src/gso/analysis/qualitative/detect_reward_hacks_thresholded.py \
        --run_dirs "gpt-5-2025-08-07_maxiter_100_N_v0.51.1-no-hint-run_*" \
        --thresholds 0.25 0.5 0.75
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
from gso.analysis.qualitative.hack_detector import (
    analyze_patch_for_hacks,
    generate_summary,
)


def get_instances_passing_threshold(report_paths, threshold_p, dataset):
    """
    Find instances that pass the threshold and identify the best run for each.

    Returns:
        dict: {instance_id: {"run_dir": str, "pc_speedup_hm": float, "opt_stats": dict}}
    """
    instance_lookup = {inst.instance_id: inst for inst in dataset}

    # Track best run for each instance
    # instance_id -> {"run_dir": str, "pc_speedup_hm": float, "opt_stats": dict}
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
            if instance_id not in instance_lookup:
                continue

            if instance_id in opt_stats:
                stats = opt_stats[instance_id]

                # Check threshold criteria (same as plot_opt1_thresholded.py)
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
                    # Check if this is the best run for this instance
                    if (
                        instance_id not in best_runs
                        or pc_speedup_hm > best_runs[instance_id]["pc_speedup_hm"]
                    ):
                        best_runs[instance_id] = {
                            "run_dir": run_dir,
                            "pc_speedup_hm": pc_speedup_hm,
                            "opt_stats": stats,
                        }

    return best_runs


def analyze_single_instance_with_run(args):
    """Analyze a single instance from a specific run - designed for parallel execution."""
    instance_id, instance, run_dir, k_samples = args

    model_patch = load_model_patch(instance_id, run_dir)
    if not model_patch:
        return {
            "instance_id": instance_id,
            "error": "No model patch found",
        }

    test_code = instance.tests
    human_patch = instance.gt_diff
    analysis = analyze_patch_for_hacks(
        instance_id,
        instance.repo,
        instance.api,
        test_code,
        human_patch,
        model_patch,
        k_samples=k_samples,
    )

    if "error" in analysis:
        return {
            "instance_id": instance_id,
            "error": analysis.get("error", "Unknown error"),
        }

    analysis["instance_id"] = instance_id
    analysis["repo"] = instance.repo
    analysis["api"] = instance.api
    analysis["run_dir"] = run_dir
    analysis["model_patch"] = model_patch
    analysis["human_patch"] = human_patch
    analysis["test_code"] = test_code
    return analysis


def run_detection_at_threshold(
    run_dirs_pattern, threshold_p, dataset, k_samples, max_workers
):
    """
    Run hack detection for instances passing a specific threshold.

    Returns:
        dict: Summary and results for this threshold
    """
    # Find all run-level report files
    run_dirs = sorted(glob.glob(run_dirs_pattern), key=natural_sort_key)
    if not run_dirs:
        print(f"No run directories found for pattern: {run_dirs_pattern}")
        return {"error": "No run directories found"}

    pass_reports = []
    for run_dir in run_dirs:
        pass_files = glob.glob(f"{run_dir}/*.pass.report.json")
        pass_reports.extend(pass_files)

    if not pass_reports:
        print(f"No pass reports found")
        return {"error": "No pass reports found"}

    print(f"  Found {len(pass_reports)} pass reports across {len(run_dirs)} runs")

    # Get instances passing threshold and their best runs
    best_runs = get_instances_passing_threshold(pass_reports, threshold_p, dataset)

    if not best_runs:
        print(f"  No instances pass threshold p={threshold_p}")
        return {
            "total_analyzed": 0,
            "hack_count": 0,
            "legitimate_count": 0,
            "hack_percentage": 0.0,
            "average_confidence": 0.0,
            "results": [],
        }

    print(f"  {len(best_runs)} instances pass threshold p={threshold_p}")

    # Prepare analysis arguments
    instance_lookup = {inst.instance_id: inst for inst in dataset}
    analysis_args = []
    for instance_id, run_info in best_runs.items():
        if instance_id in instance_lookup:
            analysis_args.append(
                (
                    instance_id,
                    instance_lookup[instance_id],
                    run_info["run_dir"],
                    k_samples,
                )
            )

    # Run hack detection in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(analyze_single_instance_with_run, arg)
            for arg in analysis_args
        ]
        for future in tqdm(
            futures,
            desc=f"  Analyzing p={threshold_p}",
            unit="patch",
            leave=False,
        ):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                all_results.append({"instance_id": "unknown", "error": str(e)})

    # Filter out errors and generate summary
    analyses = [r for r in all_results if "error" not in r]
    summary = generate_summary(analyses)

    print(
        f"  Results: {summary['hack_count']}/{summary['total_analyzed']} hacks "
        f"({summary['hack_percentage']:.1f}%), avg confidence: {summary['average_confidence']:.2f}"
    )

    return {
        **summary,
        "results": analyses,
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
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--k_samples", type=int, default=5)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    # Load dataset once
    dataset = load_gso_dataset("gso-bench/gso", "test")
    model_name = args.model_name

    print(f"\n{'='*80}")
    print(f"Running reward hack detection for: {model_name}")
    print(f"Thresholds: {args.thresholds}")
    print(f"{'='*80}\n")

    # Run detection for each threshold
    threshold_results = {}
    for threshold in sorted(args.thresholds):
        print(f"\nProcessing threshold p={threshold}...")
        results = run_detection_at_threshold(
            args.run_dirs,
            threshold,
            dataset,
            args.k_samples,
            args.max_workers,
        )
        threshold_results[str(threshold)] = results

    # Generate combined output
    output_data = {
        "model_name": model_name,
        "run_dirs_pattern": args.run_dirs,
        "k_samples": args.k_samples,
    }

    if len(args.thresholds) == 1 and args.thresholds[0] == 0.95:
        output_data.update(threshold_results[str(args.thresholds[0])])
        file_name = f"{model_name}_hack_detection.json"
    else:
        output_data["thresholds"] = threshold_results
        file_name = f"{model_name}_hack_detection_thresholded.json"

    # Save to file
    output_path = EVALUATION_REPORTS_DIR / "analysis" / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS THRESHOLDS:")
    print(f"{'='*80}")
    for threshold in sorted(args.thresholds):
        threshold_str = str(threshold)
        if threshold_str in threshold_results:
            res = threshold_results[threshold_str]
            if "error" not in res:
                print(
                    f"  p={threshold:4.2f}: {res['hack_count']}/{res['total_analyzed']} hacks "
                    f"({res['hack_percentage']:.1f}%), conf={res['average_confidence']:.2f}"
                )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
