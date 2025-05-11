import subprocess
import re
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import random
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
import glob

# Import the helper function from the original script
from pyperf.harness.utils import natural_sort_key
from pyperf.harness.opt_at_k import merge_reports
from pyperf.harness.scripts.helpers import *


def main():
    MAX_K = 5
    FIXED_FIRST_RUN = False  # Using the same as in the example
    NUM_TRIALS = 1000
    OUTPUT_DIR = "plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the report patterns
    report_patterns = [
        # "~/pyperf/reports/v0.25.0/claude-3-5-sonnet-v2-20241022_maxiter_50_N_v0.25.0-no-hint-detailed_plans-*",
        # "~/pyperf/reports/v0.25.0/claude-3-5-sonnet-v2-20241022_maxiter_50_N_v0.25.0-no-hint-run_*",
        "~/pyperf/reports/o4-mini_maxiter_100_N_v0.35.0-no-hint-detailed_plans-*planned*",
        "~/pyperf/reports/o4-mini_maxiter_100_N_v0.35.0-no-hint-run_*",
    ]

    # Create labels for the plots (extracting meaningful parts from the pattern)
    labels = ["Opt@K w/ gt plan", "Opt@K w/o gt plan"]

    # Expand the glob patterns and sort the files
    all_reports = []
    for pattern in report_patterns:
        reports = sorted(glob.glob(os.path.expanduser(pattern)), key=natural_sort_key)
        all_reports.append(reports)

    # Calculate Opt@K for both report sets
    results = []
    for reports in all_reports:
        _, _, commit_at_k_rates, _ = calculate_opt_at_k_smooth(
            reports, MAX_K, FIXED_FIRST_RUN, NUM_TRIALS
        )
        results.append(commit_at_k_rates)

    # Create DataFrame for plotting
    k_values = list(range(1, MAX_K + 1))
    plot_data = []

    # Add data for both report sets (commit metric only)
    for i, (report_results, label) in enumerate(zip(results, labels)):
        for k, rates in enumerate(report_results, 1):
            error = rates[1] if k < MAX_K else 0
            plot_data.append(
                {
                    "k": k,
                    "Rate": rates[0],
                    "Error": error,
                    "Report Set": label,
                    "Lower": rates[0] - error,
                    "Upper": rates[0] + error,
                }
            )

    df_plot = pd.DataFrame(plot_data)
    print(df_plot)
    colors = METRICS_COLOR_MAP
    plt.figure(figsize=(5, 3))
    setup_plot_style()

    # Plot lines
    sns.lineplot(
        data=df_plot,
        x="k",
        y="Rate",
        hue="Report Set",
        style="Report Set",
        markers={
            "Opt@K w/ gt plan": "o",
            "Opt@K w/o gt plan": "o",
        },
        markeredgewidth=0,
        markersize=5,
        dashes=False,
        palette=colors,
    )

    # Add error bands
    for label, color in colors.items():
        label_data = df_plot[df_plot["Report Set"] == label]
        plt.fill_between(
            label_data["k"],
            label_data["Lower"],
            label_data["Upper"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

    # Add value labels
    for label, color in colors.items():
        label_data = df_plot[df_plot["Report Set"] == label]
        for _, row in label_data.iterrows():
            plt.annotate(
                f'{row["Rate"]:.1f}%',
                (row["k"], row["Rate"]),
                textcoords="offset points",
                xytext=(0, 15) if label == "Opt@K w/ gt plan" else (0, -15),
                ha="center",
                va="top" if label == "Opt@K w/ gt plan" else "bottom",
                color="#3b3b3b",
                fontsize=12,
            )

    # Customize plot
    plt.tick_params(axis="both", direction="out", length=3, width=1)
    plt.xlabel("# Rollouts (K)")
    plt.ylabel("Opt@K (%)")
    plt.xticks(k_values)
    plt.ylim(0, 25)
    plt.grid(False)  #  linestyle="-", alpha=0.005)
    plt.legend(title=None)

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, "opt_at_k.backtranslate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")


if __name__ == "__main__":
    main()
