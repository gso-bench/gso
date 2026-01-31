import matplotlib.pyplot as plt
import os
import glob
import json
import numpy as np
import random
from pathlib import Path

from gso.harness.utils import natural_sort_key
from gso.analysis.quantitative.helpers import *
from gso.constants import PLOTS_DIR, EVALUATION_REPORTS_DIR, MIN_PROB_SPEEDUP

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_CONFIGS = {
    "Gemini-3-Flash": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/gemini-3-flash-preview_maxiter_100_N_v0.51.1-no-hint-run_*",
    "GPT-5.1": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/gpt-5.1-2025-11-13_maxiter_100_N_v0.51.1-no-hint-run_*",
    "GPT-5.2": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/gpt-5.2-2025-12-11_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Opus-4.5": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/claude-opus-4-5-20251101_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Gemini-3-Pro": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/gemini-3-pro-preview_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Sonnet-4.5": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/claude-sonnet-4-5-20250929_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Gemini-2.5-Pro": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/gemini-2.5-pro_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Qwen-3-Coder": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/qwen3-coder-plus_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Kimi-K2": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/Kimi-K2-Instruct_maxiter_100_N_v0.51.1-no-hint-run_*",
    "GPT-5": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/gpt-5-2025-08-07_maxiter_100_N_v0.51.1-no-hint-run_*",
    "Opus-4": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/claude-opus-4-20250514_maxiter_100_N_v0.51.1-no-hint-run_*",
    "o3": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/o3_maxiter_100_N_v0.35.0-no-hint-run_*",
    "Sonnet-4": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/claude-sonnet-4-20250514_maxiter_100_N_v0.35.0-no-hint-run_*",
    # "Claude-3.6": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/claude-3-5-sonnet-v2-20241022_maxiter_100_N_v0.35.0-no-hint-run_*",
    # "o4-mini": "/home/gcpuser/gso-internal/logs/run_evaluation/pass/o4-mini_maxiter_100_N_v0.35.0-no-hint-run_*",
}

# Threshold values for p (speedup thresholds)
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
NUM_TRIALS = 500
FIGSIZE = (7, 4)
MARKER_SIZE = 4
Y_LIMIT = 100
USE_COLOR_MAP = False
SHOW_STD = False

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_json(fname):
    """Load JSON file."""
    with open(fname, "r") as f:
        return json.load(f)


def calculate_opt_at_k_with_threshold_from_reports(
    report_paths, threshold_p, k=1, num_trials=500
):
    """
    Calculate Opt@K with a custom speedup threshold from individual run reports.
    This replicates the notebook's approach by using all attempted instances.
    """
    # Load all reports
    all_report_data = []
    for report_path in report_paths:
        with open(report_path, "r") as f:
            report = json.load(f)
        all_report_data.append(report)

    # Create id_rankings_dict structure (instance_id -> run_id -> opt_status)
    id_rankings_dict = {}
    for run_id, report in enumerate(all_report_data):
        opt_stats = report.get("opt_stats", {})
        instance_sets = report.get("instance_sets", {})

        attempted_instances = set()
        for key, instances in instance_sets.items():
            if key.endswith("_ids"):
                attempted_instances.update(instances)

        for instance_id in attempted_instances:
            if instance_id not in id_rankings_dict:
                id_rankings_dict[instance_id] = {}

            if instance_id in opt_stats:
                stats = opt_stats[instance_id]
                opt_status = True if threshold_p == 0.0 else False
                if threshold_p > 0.0:
                    base_mean = stats.get("base_mean", 0.0)
                    patch_mean = stats.get("patch_mean", 0.0)
                    pb_speedup_gm = stats.get("gm_speedup_patch_base", 0.0)
                    pc_speedup_hm = stats.get("hm_speedup_patch_commit", 0.0)
                    opt_status = (
                        base_mean > patch_mean
                        and round(pb_speedup_gm, 1) >= MIN_PROB_SPEEDUP
                        and pc_speedup_hm > threshold_p
                    )
            else:
                opt_status = False

            id_rankings_dict[instance_id][run_id] = opt_status

    total_instances = len(id_rankings_dict)
    opt_at_k_trials = np.zeros(num_trials)

    # Run bootstrap trials
    for trial in range(num_trials):
        opt_hits = 0

        # Process each instance
        for instance_id, rankings in id_rankings_dict.items():
            rankings = list(rankings.items())  # (run_id, opt_status)
            random.shuffle(rankings)

            # Check if any of the first k runs achieved optimization
            if any(r[1] for r in rankings[:k]):
                opt_hits += 1

        opt_at_k_trials[trial] = opt_hits / total_instances * 100.0

    return opt_at_k_trials.mean(), opt_at_k_trials.std()


def plot_opt1_vs_threshold(results_data, figsize=FIGSIZE):
    """Plot Opt@1 vs threshold with error bands."""
    import matplotlib.ticker as mtick

    plt.figure(figsize=figsize, dpi=150)

    for model_name, data in results_data.items():
        thresholds = [d["threshold"] for d in data]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]
        color = None
        if USE_COLOR_MAP:
            color = MODEL_COLOR_MAP[model_name.lower()]

        plt.plot(
            thresholds,
            means,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=2,
            label=model_name,
            color=color,
        )

        # Error bands
        if SHOW_STD:
            plt.fill_between(
                thresholds,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2,
            )

    ax = plt.gca()
    ax.set_xlabel("Speedup Threshold ($p$)", fontsize=14)
    ax.set_ylabel("Opt$_p$@1", fontsize=14)

    # Match the original plot styling
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.xaxis.set_minor_locator(mtick.FixedLocator([0.95]))
    ax.tick_params(
        axis="x",
        which="minor",
        bottom=True,
        labelbottom=False,
        length=3,
    )
    ax.tick_params(axis="x", which="major", rotation=0, pad=8, labelsize=12)
    ax.margins(x=0.02)

    ax.set_ylim(0, Y_LIMIT)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(
        frameon=True,
        fontsize=10 if len(MODEL_CONFIGS) > 3 else 14,
        loc="upper right",
        ncol=2 if len(MODEL_CONFIGS) > 3 else 1,
    )

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "opt1_thresholded.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")
    plt.show()


def main():
    """Main function to run the analysis."""
    results_data = {}

    for model_name, pattern in MODEL_CONFIGS.items():
        print(f"\nProcessing {model_name}...")

        # Find report files (they are in subdirectories)
        report_dirs = sorted(glob.glob(pattern), key=natural_sort_key)
        if not report_dirs:
            print(f"No report directories found for {model_name}")
            continue

        # Find pass reports in each directory
        pass_reports = []
        for report_dir in report_dirs:
            pass_files = glob.glob(f"{report_dir}/*.pass.report.json")
            pass_reports.extend(pass_files)

        if not pass_reports:
            print(f"No pass reports found for {model_name}")
            continue

        print(f"Found {len(pass_reports)} pass reports")

        # Calculate Opt@1 for each threshold
        model_results = []
        for p in P_VALUES:
            mean, std = calculate_opt_at_k_with_threshold_from_reports(
                pass_reports, threshold_p=p, k=1, num_trials=NUM_TRIALS
            )
            print(f"    Opt_{p}@1 = {mean:.2f} ± {std:.2f}")
            model_results.append({"threshold": p, "mean": mean, "std": std})

        results_data[model_name] = model_results

    if not results_data:
        print("No data to plot!")
        return

    # Plot the results
    setup_plot_style()
    plot_opt1_vs_threshold(results_data)

    # save results_data to a json file in reports directory
    results_dir = (EVALUATION_REPORTS_DIR / "analysis").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "opt1_thresholded.json", "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to {results_dir / 'opt1_thresholded.json'}")


if __name__ == "__main__":
    main()
