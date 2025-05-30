import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob
import argparse

from gso.harness.utils import natural_sort_key
from gso.harness.plot.helpers import *

# Configure model report patterns here
MODEL_CONFIGS = {
    "o4-mini": "~/gso/reports/o4-mini_maxiter_100_N_v0.35.0-no-hint-run_*scale*",
    "claude-3.6": "~/gso/reports/claude-3-5-sonnet-v2-20241022_maxiter_100_N_v0.35.0-no-hint-run_*scale*",
}

# Add argument parsing
parser = argparse.ArgumentParser(
    description="Plot Opt@K for multiple models on a single graph"
)
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

# Setup plot
plt.figure(figsize=(6, 4))
setup_plot_style()

# Process each model
all_data = []

for model_name, report_pattern in MODEL_CONFIGS.items():
    # Expand and sort report files
    reports = sorted(
        glob.glob(os.path.expanduser(report_pattern)), key=natural_sort_key
    )

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

    # Calculate Opt@K with the reference approach
    _, _, commit_at_k_rates, _ = calculate_opt_at_k_smooth(
        reports, args.k, args.fixed_first_run, args.num_trials
    )

    # Create data for plotting
    for k, rates in enumerate(commit_at_k_rates, 1):
        error = rates[1] if k < args.k else 0
        all_data.append(
            {
                "k": k,
                "Rate": rates[0],
                "Error": error,
                "Model": model_name,
                "Lower": rates[0] - error,
                "Upper": rates[0] + error,
            }
        )

# Create dataframe for plotting
df_plot = pd.DataFrame(all_data)

print(df_plot)

# Plot the data
sns.lineplot(
    data=df_plot,
    x="k",
    y="Rate",
    hue="Model",
    style="Model",
    markers=True,
    markeredgewidth=0,
    markersize=5,
    dashes=False,
    palette={
        model: MODEL_COLOR_MAP.get(model, "#999999")
        for model in df_plot["Model"].unique()
    },
)

# Add error bands
for model in df_plot["Model"].unique():
    model_data = df_plot[df_plot["Model"] == model]
    color = MODEL_COLOR_MAP.get(model, "#999999")
    plt.fill_between(
        model_data["k"],
        model_data["Lower"],
        model_data["Upper"],
        alpha=0.2,
        color=color,
        linewidth=0,
    )

# Add value labels at specific k points
models = df_plot["Model"].unique()
k_values = list(range(1, args.k + 1))
annotation_ks = [2, 4, 6, 8, 10]

# Pivot the dataframe so we can easily look up the other model’s rate at the same k
rates_by_k = df_plot.pivot(index="k", columns="Model", values="Rate")

for model in df_plot["Model"].unique():
    model_data = df_plot[df_plot["Model"] == model].sort_values("k")

    for _, row in model_data.iterrows():
        k = row["k"]
        if k not in annotation_ks:
            continue

        y = row["Rate"]
        # find the other model name
        other = [m for m in df_plot["Model"].unique() if m != model][0]
        other_y = rates_by_k.loc[k, other]

        # if this model’s line is above the other at this k, put the label above;
        # otherwise put it below
        offset = 8  # how many points to shift
        if y > other_y:
            xytext = (0, offset)
            va = "bottom"
        else:
            xytext = (0, -offset)
            va = "top"

        plt.annotate(
            f"{y:.1f}",
            (k, y),
            textcoords="offset points",
            xytext=xytext,
            ha="center",
            va=va,
            color="#3b3b3b",
            fontsize=12,
        )

# Customize plot
plt.tick_params(axis="both", direction="out", length=3, width=1)
plt.xlabel("# Rollouts (K)")
plt.ylabel("Opt@K (%)")
plt.xticks(k_values)
plt.ylim(0, 25)  # Adjust as needed
plt.grid(False)
plt.legend(title=None, loc="upper left")

# Save the plot
output_path = os.path.join(args.output_dir, f"opt_at_k_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
