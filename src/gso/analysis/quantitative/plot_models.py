import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
import glob

from gso.harness.utils import natural_sort_key
from gso.harness.opt_at_k import merge_reports
from gso.constants import EVALUATION_REPORTS_DIR
from gso.analysis.quantitative.helpers import *

MODEL_CONFIGS = {
    "gpt-4o": "~/gso/reports/gpt-4o_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
    "o3-mini": "~/gso/reports/o3-mini_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
    "o4-mini": "~/gso/reports/o4-mini_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
    "claude-3.6": "~/gso/reports/claude-3-5-sonnet-v2-20241022_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
    "claude-3.7": "~/gso/reports/claude-3-7-sonnet-20250219_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
    "claude-4": "~/gso/reports/claude-sonnet-4-20250514_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
    # "gemini-2.5": "~/gso/reports/gemini-2.5_maxiter_100_N_v0.35.0-no-hint-run_*pass*",
}

# Add argument parsing
parser = argparse.ArgumentParser(
    description="Run and plot Opt@1 comparisons across models"
)
parser.add_argument(
    "--output_dir", type=str, default="plots", help="Directory to save plots"
)
parser.add_argument("--k", type=int, default=1, help="K value to plot (default is 1)")
parser.add_argument(
    "--fixed_first_run", action="store_true", help="Keep first run fixed across trials"
)
parser.add_argument(
    "--num_trials", type=int, default=500, help="Number of bootstrap trials"
)
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)


def create_comparison_plot(df_results):
    """
    Create a professional-looking bar chart for model comparison with
    provider-specific colors.
    """
    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()

    # Map colors to models
    bar_colors = [
        MODEL_COLOR_MAP.get(model, "#999999") for model in df_results["Model"]
    ]

    # Create the bars with model-specific colors
    bars = ax.bar(
        df_results["Model"],
        df_results["Opt@1"],
        color=bar_colors,
        yerr=df_results["Std"],
        capsize=4,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "alpha": 0.8},
    )

    # set x label rotation and size
    ax.set_xticks(range(len(df_results["Model"])))
    ax.set_xticklabels(df_results["Model"], rotation=20, ha="center", fontsize=10)

    # Add value labels on top of bars (above the error bar)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std = df_results["Std"].iloc[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    # Customize axes
    ax.set_ylabel("Opt@1 (%)")
    # ax.set_xlabel("Model")
    ax.set_ylim(0, 25)  # Adjust as needed
    ax.set_yticks(range(0, 26, 10))

    # Add subtle grid only on y-axis
    ax.grid(axis="y", linestyle="-", alpha=0.2, color="gray")

    # Better tick formatting
    ax.tick_params(axis="both", which="major", length=4, width=1)

    # Clean up the spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# Apply the styling before creating any plots
setup_plot_style()

# Create a dataframe to hold results for all models
results_data = []

# Process each model
for model_name, report_pattern in MODEL_CONFIGS.items():
    # Expand report pattern
    reports = sorted(
        glob.glob(os.path.expanduser(report_pattern)), key=natural_sort_key
    )

    # if len(reports) > 0:
    #     assert len(reports) == 3, f"Expected 3 for {model_name}, found {len(reports)}"

    # Skip if no reports found
    if not reports:
        print(f"No reports found for model {model_name}, skipping")
        continue

    # Ensure we have at least K reports
    if len(reports) < args.k:
        print(
            f"Found only {len(reports)} reports for {model_name}, need at least {args.k}"
        )
        continue

    print(f"Processing {model_name} with {len(reports)} reports...")

    # Calculate Opt@K
    passed_at_k_rates, base_at_k_rates, commit_at_k_rates, main_at_k_rates = (
        calculate_opt_at_k_smooth(
            reports, args.k, args.fixed_first_run, args.num_trials
        )
    )

    # We only need Opt@1 (index 0)
    results_data.append(
        {
            "Model": model_name,
            "Opt@1": commit_at_k_rates[0][0],  # Mean
            "Std": commit_at_k_rates[0][1],  # Standard deviation
        }
    )

# Convert to DataFrame
df_results = pd.DataFrame(results_data)
print(df_results)
fig = create_comparison_plot(df_results)
output_path = os.path.join(args.output_dir, "model_comparison.png")
fig.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"Plot saved as {output_path}")
