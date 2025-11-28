"""
Simple script to plot hacking rates across thresholds for multiple models.

This script loads hack detection reports and plots the percentage of hacks
across different speedup thresholds, similar to plot_opt1_threshold.py.
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from gso.analysis.quantitative.helpers import setup_plot_style
from gso.constants import EVALUATION_REPORTS_DIR, PLOTS_DIR

# Model configurations for thresholded reports
MODEL_REPORTS_THRESHOLDED = {
    "GPT-5": "gpt-5_hack_detection_thresholded.json",
    "o3": "o3_hack_detection_thresholded.json",
    "Sonnet-4.5": "claude-sonnet-4.5_hack_detection_thresholded.json",
    "Gemini-2.5-Pro": "gemini-2.5-pro_hack_detection_thresholded.json",
    "Qwen-3-Coder": "qwen3-coder_hack_detection_thresholded.json",
    "Kimi-K2": "kimi-k2_hack_detection_thresholded.json",
    "GLM-4.5-Air": "glm-4.5-air_hack_detection_thresholded.json",
}

# Model configurations for no_threshold reports
MODEL_REPORTS_NO_THRESHOLD = {
    "GPT-5": "gpt-5_hack_detection_no_threshold.json",
    "o3": "o3_hack_detection_no_threshold.json",
    "Sonnet-4": "claude-sonnet-4_hack_detection_no_threshold.json",
    "Sonnet-4.5": "claude-sonnet-4.5_hack_detection_no_threshold.json",
    "Gemini-2.5-Pro": "gemini-2.5-pro_hack_detection_no_threshold.json",
    "Qwen-3-Coder": "qwen3-coder_hack_detection_no_threshold.json",
    "Kimi-K2": "kimi-k2_hack_detection_no_threshold.json",
    "GLM-4.5-Air": "glm-4.5-air_hack_detection_no_threshold.json",
    "Opus-4.5": "claude-opus-4.5_hack_detection_no_threshold.json",
    "Gemini-3-Pro": "gemini-3-pro_hack_detection_no_threshold.json",
}

# Plot configuration
TOTAL_DATASET_SIZE = 102
FIGSIZE = (7, 4)
MARKER_SIZE = 4
Y_LIMIT = 12
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def extract_hack_rates(data):
    """Extract hack rates for each threshold from the report data."""
    if "thresholds" not in data:
        print("Warning: No thresholds data found")
        return {}
    hack_rates = {}
    for threshold_str, threshold_data in data["thresholds"].items():
        threshold = float(threshold_str)
        if threshold not in THRESHOLDS:
            continue
        if "hack_count" in threshold_data:
            # Calculate hack rate as percentage of total dataset
            hack_count = threshold_data["hack_count"]
            hack_rate = (hack_count / TOTAL_DATASET_SIZE) * 100
            hack_rates[threshold] = hack_rate
        else:
            print(f"Warning: No hack_count for threshold {threshold}")

    return hack_rates


def extract_hack_rates_no_threshold(data):
    """Extract hack rates from no_threshold data, split by is_correct."""
    if "all_analyses" not in data:
        print("Warning: No all_analyses data found")
        return {"correct": 0, "incorrect": 0}

    correct_hacks = 0
    incorrect_hacks = 0

    for analysis in data["all_analyses"]:
        if analysis.get("is_reward_hack", False):
            if analysis.get("is_correct", False):
                correct_hacks += 1
            else:
                incorrect_hacks += 1

    return {"correct": correct_hacks, "incorrect": incorrect_hacks}


def plot_hack_rates_vs_threshold(results_data):
    """Plot hack rates vs threshold for all models."""
    plt.figure(figsize=FIGSIZE, dpi=150)

    for model_name, hack_rates in results_data.items():
        thresholds = sorted(hack_rates.keys())
        rates = [hack_rates[t] for t in thresholds]

        plt.plot(
            thresholds,
            rates,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=2,
            label=model_name,
        )

    ax = plt.gca()
    ax.set_xlabel("Speedup Threshold ($p$)", fontsize=14)
    ax.set_ylabel("Failures (%)", fontsize=14)

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
        fontsize=10 if len(results_data) > 3 else 14,
        loc="upper right",
        ncol=2 if len(results_data) > 3 else 1,
    )

    plt.tight_layout()

    # Save plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "hack_rates_thresholded.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")
    plt.show()


def plot_hack_rates_no_threshold(results_data):
    """Plot unthresholded hack rates as stacked bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Sort models by total hack rate (descending)
    model_items = list(results_data.items())
    model_items.sort(
        key=lambda x: (x[1]["correct"] + x[1]["incorrect"]) / TOTAL_DATASET_SIZE,
        reverse=True,
    )
    model_names = [item[0] for item in model_items]
    correct_rates = []
    incorrect_rates = []

    for model_name in model_names:
        counts = results_data[model_name]
        correct_rate = (counts["correct"] / TOTAL_DATASET_SIZE) * 100
        incorrect_rate = (counts["incorrect"] / TOTAL_DATASET_SIZE) * 100
        correct_rates.append(correct_rate)
        incorrect_rates.append(incorrect_rate)

    x_pos = range(len(model_names))
    width = 0.6

    # Create stacked bars
    bars_correct = ax.bar(
        x_pos,
        correct_rates,
        width,
        label="Reward Hacks (Judge Detected)",
        color="#0066C0",
    )
    bars_incorrect = ax.bar(
        x_pos,
        incorrect_rates,
        width,
        bottom=correct_rates,
        label="Functionally Incorrect (Tests Failed)",
        color="#009DE4",
    )

    # ax.set_xlabel("Model", fontsize=16)
    ax.set_ylabel("Failures (%)", fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=30, ha="center", fontsize=13)
    max_rate = max(
        (correct_rates[i] + incorrect_rates[i] for i in range(len(model_names))),
        default=0,
    )
    ax.set_ylim(0, max_rate * 1.1 if max_rate > 0 else 10)
    ax.grid(True, linestyle=":", linewidth=0.5, axis="y")
    ax.legend(frameon=True, fontsize=16, loc="upper right")

    # Format y-axis as percentage (values are already in percentage, so use FuncFormatter)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.1f}%"))

    plt.tight_layout()

    # Save plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path_png = PLOTS_DIR / "hack_rates_no_threshold.png"
    output_path_svg = PLOTS_DIR / "hack_rates_no_threshold.svg"
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    print(f"Plot saved as {output_path_png}")
    print(f"Plot saved as {output_path_svg}")
    plt.show()


def main():
    """Main function to load data and create plot."""
    parser = argparse.ArgumentParser(description="Plot hack detection rates")
    parser.add_argument(
        "--mode",
        choices=["thresholded", "no_threshold"],
        default="thresholded",
        help="Plot mode: thresholded (line plot) or no_threshold (stacked bar plot)",
    )
    args = parser.parse_args()

    print(f"Loading hack detection reports (mode: {args.mode})...")

    results_data = {}

    if args.mode == "thresholded":
        model_reports = MODEL_REPORTS_THRESHOLDED
        subdir = "thresholded"
    else:
        model_reports = MODEL_REPORTS_NO_THRESHOLD
        subdir = "no_threshold"

    for model_name, report_file in model_reports.items():
        print(f"Processing {model_name}...")

        report_path = (
            EVALUATION_REPORTS_DIR
            / "analysis"
            / "hack_detection"
            / subdir
            / report_file
        )
        if not report_path.exists():
            print(f"No report found for {model_name} at {report_path}")
            continue

        with open(report_path, "r") as f:
            data = json.load(f)

        if args.mode == "thresholded":
            hack_rates = extract_hack_rates(data)
            if hack_rates:
                results_data[model_name] = hack_rates
                print(f"  Found hack rates for {len(hack_rates)} thresholds")
            else:
                print(f"  No hack rates found")
        else:
            hack_counts = extract_hack_rates_no_threshold(data)
            total_hacks = hack_counts["correct"] + hack_counts["incorrect"]
            if total_hacks > 0:
                results_data[model_name] = hack_counts
                print(
                    f"  Found {total_hacks} hacks ({hack_counts['correct']} correct, "
                    f"{hack_counts['incorrect']} incorrect)"
                )
            else:
                print(f"  No hacks found")

    if not results_data:
        print("No data to plot!")
        return

    # Calculate and report averages
    if args.mode == "no_threshold":
        total_hack_rates = []
        correct_hack_rates = []
        incorrect_hack_rates = []

        for model_name, counts in results_data.items():
            total_rate = (
                (counts["correct"] + counts["incorrect"]) / TOTAL_DATASET_SIZE
            ) * 100
            correct_rate = (counts["correct"] / TOTAL_DATASET_SIZE) * 100
            incorrect_rate = (counts["incorrect"] / TOTAL_DATASET_SIZE) * 100

            total_hack_rates.append(total_rate)
            correct_hack_rates.append(correct_rate)
            incorrect_hack_rates.append(incorrect_rate)

        avg_total = (
            sum(total_hack_rates) / len(total_hack_rates) if total_hack_rates else 0
        )
        avg_correct = (
            sum(correct_hack_rates) / len(correct_hack_rates)
            if correct_hack_rates
            else 0
        )
        avg_incorrect = (
            sum(incorrect_hack_rates) / len(incorrect_hack_rates)
            if incorrect_hack_rates
            else 0
        )

        print(f"\nAverage hack rates across {len(results_data)} models:")
        print(f"  Total: {avg_total:.2f}%")
        print(f"  Passed Tests: {avg_correct:.2f}%")
        print(f"  Failed Tests: {avg_incorrect:.2f}%")

    print(f"\nPlotting hack rates for {len(results_data)} models...")
    setup_plot_style()

    if args.mode == "thresholded":
        plot_hack_rates_vs_threshold(results_data)
        # Save results to JSON
        results_dir = EVALUATION_REPORTS_DIR / "analysis"
        with open(results_dir / "hack_rates_thresholded.json", "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_dir / 'hack_rates_thresholded.json'}")
    else:
        plot_hack_rates_no_threshold(results_data)
        # Save results to JSON
        results_dir = EVALUATION_REPORTS_DIR / "analysis"
        with open(results_dir / "hack_rates_no_threshold.json", "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {results_dir / 'hack_rates_no_threshold.json'}")


if __name__ == "__main__":
    main()
