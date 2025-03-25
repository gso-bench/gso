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
from tqdm import tqdm
import argparse

from pyperf.harness.utils import natural_sort_key
from pyperf.harness.beat_at_k import merge_reports
from pyperf.constants import EVALUATION_REPORTS_DIR

PLOT_MAIN = False

# Add argument parsing
parser = argparse.ArgumentParser(description="Run and plot beat@K evaluations")
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


def calculate_beat_at_k_smooth(report_paths, N, fixed_first_run=False, num_trials=500):
    """
    Calculate beat@k rates with bootstrapping, closely following the reference implementation.
    """
    # Load all reports into a more usable format
    all_report_data = []
    for i, report_path in enumerate(report_paths):
        with open(report_path, "r") as f:
            report = json.load(f)
        all_report_data.append(report)

    # Create id_rankings_dict structure (instance_id -> run_id -> metrics)
    id_rankings_dict = {}
    for run_id, report in enumerate(all_report_data):
        # Extract all instance IDs that have improvement metrics
        improved_base_ids = set(report["instance_sets"].get("improved_base_ids", []))
        improved_commit_ids = set(
            report["instance_sets"].get("improved_commit_ids", [])
        )
        improved_main_ids = set(report["instance_sets"].get("improved_main_ids", []))

        # All instance IDs: (just take any that appear in any of the classes)
        all_instance_ids = set()
        for key in report["instance_sets"]:
            if key.endswith("_ids"):
                all_instance_ids.update(report["instance_sets"][key])

        for instance_id in all_instance_ids:
            if instance_id not in id_rankings_dict:
                id_rankings_dict[instance_id] = {}

            # Store (improved_over_base, improved_over_commit, improved_over_main)
            id_rankings_dict[instance_id][run_id] = (
                instance_id in improved_base_ids,
                instance_id in improved_commit_ids,
                instance_id in improved_main_ids,
            )

        # assert len(id_rankings_dict) == 122, f"Expected 122 instances"

    total_instances = len(id_rankings_dict)
    base_at_n_trials = np.zeros((num_trials, N))
    commit_at_n_trials = np.zeros((num_trials, N))
    main_at_n_trials = np.zeros((num_trials, N))

    # Run multiple trials
    for trial in range(num_trials):
        base_at_n = np.zeros(N)
        commit_at_n = np.zeros(N)
        main_at_n = np.zeros(N)

        # Process each instance
        for instance_id, rankings in id_rankings_dict.items():
            rankings = list(rankings.items())  # (run_id, (base, commit, main))
            if not fixed_first_run:
                random.shuffle(rankings)

            # Process each N value
            for idx in range(N):
                # Shuffle for the second run (so we keep the first run perf unchanged)
                if fixed_first_run and idx == 1:
                    rankings = list(rankings)  # Make sure it's a list
                    random.shuffle(rankings)  # Shuffle for this trial

                n_rankings = rankings[: idx + 1]

                # Check if beat base in any
                if any(r[1][0] for r in n_rankings):
                    base_at_n[idx] += 1

                # Check if beat commit in any
                if any(r[1][1] for r in n_rankings):
                    commit_at_n[idx] += 1

                # Check if beat main in any
                if any(r[1][2] for r in n_rankings):
                    main_at_n[idx] += 1

        # Store results for this trial
        base_at_n_trials[trial] = base_at_n
        commit_at_n_trials[trial] = commit_at_n
        main_at_n_trials[trial] = main_at_n

    # Calculate means and standard deviations
    base_at_n_mean = np.mean(base_at_n_trials, axis=0)
    base_at_n_std = np.std(base_at_n_trials, axis=0)
    commit_at_n_mean = np.mean(commit_at_n_trials, axis=0)
    commit_at_n_std = np.std(commit_at_n_trials, axis=0)
    main_at_n_mean = np.mean(main_at_n_trials, axis=0)
    main_at_n_std = np.std(main_at_n_trials, axis=0)

    # Convert to percentages
    base_at_n_mean_pct = [x / total_instances * 100 for x in base_at_n_mean]
    base_at_n_std_pct = [x / total_instances * 100 for x in base_at_n_std]
    commit_at_n_mean_pct = [x / total_instances * 100 for x in commit_at_n_mean]
    commit_at_n_std_pct = [x / total_instances * 100 for x in commit_at_n_std]
    main_at_n_mean_pct = [x / total_instances * 100 for x in main_at_n_mean]
    main_at_n_std_pct = [x / total_instances * 100 for x in main_at_n_std]

    # Format results for plotting
    base_at_k_rates = list(zip(base_at_n_mean_pct, base_at_n_std_pct))
    commit_at_k_rates = list(zip(commit_at_n_mean_pct, commit_at_n_std_pct))
    main_at_k_rates = list(zip(main_at_n_mean_pct, main_at_n_std_pct))

    return base_at_k_rates, commit_at_k_rates, main_at_k_rates


# Calculate beat@k with the reference approach
base_at_k_rates, commit_at_k_rates, main_at_k_rates = calculate_beat_at_k_smooth(
    reports, args.k, args.fixed_first_run, args.num_trials
)

# Create a DataFrame in long format for seaborn (directly from reference)
k_values = list(range(1, args.k + 1))
plot_data = []

# Add data for base
for k, rates in enumerate(base_at_k_rates, 1):
    plot_data.append(
        {
            "k": k,
            "Rate": rates[0],
            "Error": rates[1],
            "Metric": "beat(base)@k",
            "Lower": rates[0] - rates[1],
            "Upper": rates[0] + rates[1],
        }
    )

# Add data for commit
for k, rates in enumerate(commit_at_k_rates, 1):
    plot_data.append(
        {
            "k": k,
            "Rate": rates[0],
            "Error": rates[1],
            "Metric": "beat(commit)@k",
            "Lower": rates[0] - rates[1],
            "Upper": rates[0] + rates[1],
        }
    )

# Add data for main if needed
if PLOT_MAIN:
    for k, rates in enumerate(main_at_k_rates, 1):
        plot_data.append(
            {
                "k": k,
                "Rate": rates[0],
                "Error": rates[1],
                "Metric": "beat(main)@k",
                "Lower": rates[0] - rates[1],
                "Upper": rates[0] + rates[1],
            }
        )

df_plot = pd.DataFrame(plot_data)

# Set the style (from reference)
sns.set_style("whitegrid")

# Define colors
colors = {
    "beat(base)@k": "#1f77b4",  # blue
    "beat(commit)@k": "#ff7f0e",  # orange
    "beat(main)@k": "#2ca02c",  # green
}

plt.figure(figsize=(8, 6))
plt.rcParams.update({"font.size": 14})

# Create the plot (closely following reference)
sns.lineplot(
    data=df_plot,
    x="k",
    y="Rate",
    hue="Metric",
    style="Metric",
    markers=True,
    dashes=False,
    palette=colors,
)

# Add error bands (from reference)
for metric, color in colors.items():
    if metric == "beat(main)@k" and not PLOT_MAIN:
        continue
    metric_data = df_plot[df_plot["Metric"] == metric]
    plt.fill_between(
        metric_data["k"],
        metric_data["Lower"],
        metric_data["Upper"],
        alpha=0.2,
        color=color,
    )

# Add value labels (from reference)
for metric, color in colors.items():
    if metric == "beat(main)@k" and not PLOT_MAIN:
        continue
    metric_data = df_plot[df_plot["Metric"] == metric]
    for _, row in metric_data.iterrows():
        plt.annotate(
            f'{row["Rate"]:.1f}%',
            (row["k"], row["Rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=color,
            fontsize=9,
        )

# Customize plot
plt.xlabel("# Agent Rollouts (K)")
plt.ylabel("% Problems")
plt.xticks(k_values)
plt.grid(True, linestyle="-", alpha=0.05)
plt.legend(title=None)

# Save the plot
output_path = os.path.join(args.output_dir, f"beat_at_k.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
