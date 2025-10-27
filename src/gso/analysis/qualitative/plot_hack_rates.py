"""
Simple script to plot hacking rates across thresholds for multiple models.

This script loads hack detection reports and plots the percentage of hacks
across different speedup thresholds, similar to plot_opt1_threshold.py.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from gso.analysis.quantitative.helpers import setup_plot_style
from gso.constants import EVALUATION_REPORTS_DIR, PLOTS_DIR

# Model configurations
MODEL_REPORTS = {
    "GPT-5": "gpt-5_hack_detection_thresholded.json",
    "o3": "o3_hack_detection_thresholded.json",
    "Sonnet-4.5": "claude-sonnet-4.5_hack_detection_thresholded.json",
    "Gemini-2.5-Pro": "gemini-2.5-pro_hack_detection_thresholded.json",
    "Qwen-3-Coder": "qwen3-coder_hack_detection_thresholded.json",
    "Kimi-K2": "kimi-k2_hack_detection_thresholded.json",
    "GLM-4.5-Air": "glm-4.5-air_hack_detection_thresholded.json",
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
    ax.set_ylabel("Hack Rate (%)", fontsize=14)

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


def main():
    """Main function to load data and create plot."""
    print("Loading hack detection reports...")

    results_data = {}

    for model_name, report_file in MODEL_REPORTS.items():
        print(f"Processing {model_name}...")

        report_path = EVALUATION_REPORTS_DIR / "analysis" / report_file
        if not report_path.exists():
            print("No report found for {model_name}")
            continue

        with open(report_path, "r") as f:
            data = json.load(f)

        hack_rates = extract_hack_rates(data)
        if hack_rates:
            results_data[model_name] = hack_rates
            print(f"  Found hack rates for {len(hack_rates)} thresholds")
        else:
            print(f"  No hack rates found")

    if not results_data:
        print("No data to plot!")
        return

    print(f"\nPlotting hack rates for {len(results_data)} models...")
    setup_plot_style()
    plot_hack_rates_vs_threshold(results_data)

    # Save results to JSON
    results_dir = EVALUATION_REPORTS_DIR / "analysis"
    with open(results_dir / "hack_rates_thresholded.json", "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Results saved to {results_dir / 'hack_rates_thresholded.json'}")


if __name__ == "__main__":
    main()
