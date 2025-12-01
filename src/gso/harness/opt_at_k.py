import json
import subprocess
from pathlib import Path
import argparse
import glob

from gso.harness.utils import natural_sort_key
from gso.constants import EVALUATION_REPORTS_DIR
from gso.utils.io import load_gso_dataset


def run_evaluation(
    pred_path,
    dataset_name,
    timeout,
    run_id,
    reformat_reports=False,
    max_workers=10,
    instance_ids=None,
    rerun_all=False,
    verbose=False,
):
    """Run evaluation script and return path to generated report."""
    cmd = [
        "uv",
        "run",
        "src/gso/harness/run_evaluation.py",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        pred_path,
        "--timeout",
        str(timeout),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
    ]

    if reformat_reports:
        cmd.append("--reformat_reports")

    if instance_ids:
        cmd.extend(["--instance_ids", *instance_ids])

    if rerun_all:
        cmd.append("--rerun_all")

    if verbose:
        cmd.append("--verbose")

    output = subprocess.check_output(cmd, text=True)
    print(output)
    report_path = next(
        line.split(": ")[1] for line in output.split("\n") if "Run Report:" in line
    )
    return report_path


def merge_reports(full_dataset, report_files, k):
    """Merge multiple report files with OR logic for pass status."""
    # Define constants and initial structure
    STATUS_PRIORITY = {
        "passed": 5,
        "test_failed": 4,
        "patch_failed": 3,
        "empty_patch": 2,
        "base_failed": 1,
        "error": 0,
        None: -1,
    }

    STATUS_SETS = [
        "completed_ids",
        "passed_ids",
        "base_failed_ids",
        "patch_failed_ids",
        "test_failed_ids",
        "empty_patch_ids",
        "error_ids",
        "opt_base_ids",
        "opt_commit_ids",
        "opt_main_ids",
    ]

    IMPROVEMENT_METRICS = [
        "opt_base_ids",
        "opt_commit_ids",
        "opt_main_ids",
    ]

    report = {
        "summary": {
            key: 0
            for key in [
                "total_instances",
                "total_predictions",
                "completed_instances",
                "incomplete_instances",
                "passed_instances",
                "base_failed_instances",
                "patch_failed_instances",
                "test_failed_instances",
                "empty_patch_instances",
                "error_instances",
                "opt_base",
                "opt_commit",
                "opt_main",
            ]
        },
        "instance_sets": {key: set() for key in STATUS_SETS},
        "schema_version": 1,
    }
    report["summary"]["k"] = k

    instance_status = {}  # instance_id -> best status across runs
    instance_opt_stats = {}  # instance_id -> best gm_speedup_patch_commit across runs

    def get_instance_status(instance_id, current_sets):
        """Determine status for an instance based on its presence in status sets."""
        status_mapping = {
            "passed_ids": "passed",
            "patch_failed_ids": "patch_failed",
            "test_failed_ids": "test_failed",
            "empty_patch_ids": "empty_patch",
            "base_failed_ids": "base_failed",
            "error_ids": "error",
        }
        for set_name, status in status_mapping.items():
            if instance_id in current_sets[set_name]:
                return status
        return None

    # Process each report file
    for report_file in report_files:
        with open(report_file) as f:
            current_report = json.load(f)
            current_sets = current_report["instance_sets"]

            # Update instance statuses
            for instance_id in current_sets["completed_ids"]:
                status = get_instance_status(instance_id, current_sets)
                current_priority = STATUS_PRIORITY.get(instance_status.get(instance_id))
                new_priority = STATUS_PRIORITY.get(status)
                if new_priority > current_priority:
                    instance_status[instance_id] = status

            # Merge improvement tracking
            for metric in IMPROVEMENT_METRICS:
                report["instance_sets"][metric].update(current_sets[metric])

    # Populate instance sets based on best status
    for instance_id, status in instance_status.items():
        report["instance_sets"]["completed_ids"].add(instance_id)
        if status:
            report["instance_sets"][f"{status}_ids"].add(instance_id)

    # Populate opt stats
    opt_commit_ids = set(report["instance_sets"]["opt_commit_ids"])
    for report_file in report_files:
        with open(report_file) as f:
            current_report = json.load(f)
            for instance_id in opt_commit_ids:
                if instance_id in current_report["opt_stats"]:
                    new_opt_stats = current_report["opt_stats"][instance_id]
                    current_opt_stats = instance_opt_stats.get(instance_id, {})

                    if new_opt_stats.get("gm_speedup_patch_commit"):
                        if (
                            not current_opt_stats
                            or not current_opt_stats.get("gm_speedup_patch_commit")
                            or new_opt_stats["gm_speedup_patch_commit"]
                            > current_opt_stats.get("gm_speedup_patch_commit", 0)
                        ):
                            new_opt_stats["report_file"] = report_file
                            instance_opt_stats[instance_id] = new_opt_stats

    report["opt_stats"] = instance_opt_stats

    # Convert sets to sorted lists
    for key in report["instance_sets"]:
        report["instance_sets"][key] = sorted(report["instance_sets"][key])

    # Update summary counts
    summary_mapping = {
        "total_instances": len(full_dataset),
        "total_predictions": len(instance_status),
        "completed_instances": "completed_ids",
        "passed_instances": "passed_ids",
        "patch_failed_instances": "patch_failed_ids",
        "test_failed_instances": "test_failed_ids",
        "empty_patch_instances": "empty_patch_ids",
        "base_failed_instances": "base_failed_ids",
        "error_instances": "error_ids",
        "opt_base": "opt_base_ids",
        "opt_commit": "opt_commit_ids",
        "opt_main": "opt_main_ids",
    }

    for summary_key, instance_set_key in summary_mapping.items():
        if isinstance(instance_set_key, str):
            report["summary"][summary_key] = len(
                report["instance_sets"][instance_set_key]
            )
        else:
            report["summary"][summary_key] = instance_set_key

    # Print summary
    summary = report["summary"]
    score = round(summary["opt_commit"] / summary["total_instances"] * 100, 2)
    report["summary"]["score"] = score

    print("\n=== Evaluation Summary ===")
    print(f"Total instances: {summary['total_instances']}")
    print(f"Instances submitted: {summary['total_predictions']}")
    print(f"Instances completed: {summary['completed_instances']}")
    print(f"Incomplete incomplete: {summary['incomplete_instances']}")
    print("-" * 10)
    print(f"Instances that passed: {summary['passed_instances']}")
    print(f"Instances with failed tests: {summary['test_failed_instances']}")
    print(f"Instances with failed patch: {summary['patch_failed_instances']}")
    print(f"Instances with empty patches: {summary['empty_patch_instances']}")
    print(f"Instances with base errors: {summary['base_failed_instances']}")
    print(f"Instances with errors: {summary['error_instances']}")
    print("-" * 10)
    print(f"Instances that improved over base: {summary['opt_base']}")
    print(f"Instances that improved over commit: {summary['opt_commit']}")
    print("=" * 10)
    print(f"score [opt@{k}]: {score}%")
    print("=" * 10)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run and merge multiple evaluation runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arguments matching run_evaluation.py
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gso-bench/gso",
        help="Name of HF dataset to use or local json/jsonl file",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout (in seconds) for running tests",
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt-4o", help="Model name for output file"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k for pass@k / Number of predictions for each instance",
    )
    parser.add_argument(
        "--prediction_paths",
        type=str,
        nargs="+",
        required=True,
        help="Space separated list of prediction paths",
    )
    parser.add_argument(
        "--reformat_reports",
        action="store_true",
        help="Reformat and rewrite reports for instances that have already been run",
    )
    parser.add_argument(
        "--rerun_all",
        action="store_true",
        help="Rerun all instances, even if they have already been run. Default: False",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Max workers for parallel processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-run summaries",
    )

    args = parser.parse_args()

    # Use the paths as provided by bash, take first k
    prediction_paths = sorted(args.prediction_paths, key=natural_sort_key)[: args.k]
    print(f"Found {len(args.prediction_paths)} predictions, using first k={args.k}:")

    # Run evaluations and collect report paths
    report_files = []
    for pred_path in prediction_paths:
        print(f"\nRunning evaluation for: {pred_path}")
        report_path = run_evaluation(
            pred_path,
            args.dataset_name,
            args.timeout,
            args.run_id,
            args.reformat_reports,
            args.max_workers,
            args.instance_ids,
            args.rerun_all,
            args.verbose,
        )
        report_files.append(report_path)

    # Merge reports
    print("\nMerging reports...")
    full_dataset = load_gso_dataset(args.dataset_name, "test")
    merged_results = merge_reports(full_dataset, report_files, args.k)

    # Save report results
    EVALUATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = EVALUATION_REPORTS_DIR / Path(
        f"{args.model_name}.opt@{args.k}.{args.run_id}.report.json"
    )
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
