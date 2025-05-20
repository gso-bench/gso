import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse

from gso.harness.utils import natural_sort_key
from gso.harness.opt_at_k import merge_reports
from gso.constants import EVALUATION_REPORTS_DIR
from gso.harness.plot.helpers import *

PLOT_PASSED = True
PLOT_HAS_SPEEDUP = True
PLOT_COMMIT = False

# Add argument parsing
parser = argparse.ArgumentParser(description="Run and plot Opt@K evaluations")
parser.add_argument(
    "--eval_reports", type=str, nargs="+", help="evaluated reports", required=True
)
parser.add_argument("--model_name", type=str, help="Model name", required=True)
parser.add_argument("--k", type=int, default=10, help="Maximum K value")
parser.add_argument(
    "--output_dir", type=str, default="plots", help="Directory to save plots"
)
parser.add_argument(
    "--fixed_first_run", action="store_true", help="Keep first run fixed across trials"
)
parser.add_argument(
    "--num_trials", type=int, default=500, help="Number of bootstrap trials"
)
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# sort report files
reports = sorted(args.eval_reports, key=natural_sort_key)

# raise if less than k reports are found
if len(reports) < args.k:
    raise ValueError(f"Found {len(reports)} reports, expected {args.k}")


# Calculate Opt@K with the reference approach
passed_at_k_rates, base_at_k_rates, commit_at_k_rates, main_at_k_rates = (
    calculate_opt_at_k_smooth(reports, args.k, args.fixed_first_run, args.num_trials)
)

# Create a DataFrame in long format for seaborn (directly from reference)
k_values = list(range(1, args.k + 1))
plot_data = []


# Add data for passed
if PLOT_PASSED:
    for k, rates in enumerate(passed_at_k_rates, 1):
        error = rates[1] if k < args.k else 0
        plot_data.append(
            {
                "k": k,
                "Rate": rates[0],
                "Error": error,
                "Metric": "Passed Tests",
                "Lower": rates[0] - error,
                "Upper": rates[0] + error,
            }
        )

# Add data for speedup on base
if PLOT_HAS_SPEEDUP:
    for k, rates in enumerate(base_at_k_rates, 1):
        error = rates[1] if k < args.k else 0
        plot_data.append(
            {
                "k": k,
                "Rate": rates[0],
                "Error": error,
                "Metric": "Has Speedup over Base",
                "Lower": rates[0] - error,
                "Upper": rates[0] + error,
            }
        )

# Add data for commit
if PLOT_COMMIT:
    for k, rates in enumerate(commit_at_k_rates, 1):
        error = rates[1] if k < args.k else 0
        plot_data.append(
            {
                "k": k,
                "Rate": rates[0],
                "Error": error,
                "Metric": "Opt@K",
                "Lower": rates[0] - error,
                "Upper": rates[0] + error,
            }
        )

df_plot = pd.DataFrame(plot_data)
colors = METRICS_COLOR_MAP
plt.figure(figsize=(8, 6))
setup_plot_style()

# Create the plot
sns.lineplot(
    data=df_plot,
    x="k",
    y="Rate",
    hue="Metric",
    style="Metric",
    markers={"Opt@K": "o", "Has Speedup over Base": "o", "Passed Tests": "o"},
    markeredgewidth=0,
    markersize=5,
    dashes=False,
    palette=colors,
)

# Add error bands
for metric, color in colors.items():
    metric_data = df_plot[df_plot["Metric"] == metric]
    plt.fill_between(
        metric_data["k"],
        metric_data["Lower"],
        metric_data["Upper"],
        alpha=0.2,
        color=color,
        linewidth=0,
    )

# Add value labels
for metric, color in colors.items():
    metric_data = df_plot[df_plot["Metric"] == metric]
    for idx, (_, row) in enumerate(metric_data.iterrows()):
        if idx in [0, 2, 4, 7, 9]:
            plt.annotate(
                f'{row["Rate"]:.1f}',
                (row["k"], row["Rate"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                color="#3b3b3b",
                fontsize=12,
            )

# Customize plot
plt.tick_params(axis="both", direction="out", length=3, width=1)
plt.xlabel("# Rollouts (K)")
plt.ylabel("% Problems")
plt.xticks(k_values)
plt.ylim(-3, 101)
plt.grid(False)  #  linestyle="-", alpha=0.005)
plt.legend(title=None, loc="lower right")

# Save the plot
output_path = os.path.join(args.output_dir, f"opt_at_k.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
