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
from pyperf.harness.beat_at_k import merge_reports


# Use the same function from the original script
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
        beat_base_ids = set(report["instance_sets"].get("beat_base_ids", []))
        beat_commit_ids = set(report["instance_sets"].get("beat_commit_ids", []))
        beat_main_ids = set(report["instance_sets"].get("beat_main_ids", []))

        # All instance IDs: (just take any that appear in any of the classes)
        all_instance_ids = set()
        for key in report["instance_sets"]:
            if key.endswith("_ids"):
                all_instance_ids.update(report["instance_sets"][key])

        for instance_id in all_instance_ids:
            if instance_id not in id_rankings_dict:
                id_rankings_dict[instance_id] = {}

            # Store (beat_base, beat_commit, beat_main)
            id_rankings_dict[instance_id][run_id] = (
                instance_id in beat_base_ids,
                instance_id in beat_commit_ids,
                instance_id in beat_main_ids,
            )

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


def main():
    MAX_K = 5
    FIXED_FIRST_RUN = False  # Using the same as in the example
    NUM_TRIALS = 1000
    OUTPUT_DIR = "plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the report patterns
    report_patterns = [
        # "~/pyperf/reports/claude-3-5-sonnet-v2-20241022_maxiter_50_N_v0.25.0-no-hint-short_plans-*",
        "~/pyperf/reports/claude-3-5-sonnet-v2-20241022_maxiter_50_N_v0.25.0-no-hint-detailed_plans-*",
        "~/pyperf/reports/claude-3-5-sonnet-v2-20241022_maxiter_50_N_v0.25.0-no-hint-run_*",
    ]

    # Create labels for the plots (extracting meaningful parts from the pattern)
    labels = ["Beat@K w/ detailed plan", "Beat@K w/o gt plan"]

    # Expand the glob patterns and sort the files
    all_reports = []
    for pattern in report_patterns:
        reports = sorted(glob.glob(os.path.expanduser(pattern)), key=natural_sort_key)
        all_reports.append(reports)

    # Calculate beat@k for both report sets
    results = []
    for reports in all_reports:
        _, commit_at_k_rates, _ = calculate_beat_at_k_smooth(
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

    # Define colors for the two lines
    colors = {
        labels[0]: "#3d9651",  # green
        labels[1]: "#da7c2f",  # orange
    }

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 14})

    # Plot lines
    sns.lineplot(
        data=df_plot,
        x="k",
        y="Rate",
        hue="Report Set",
        style="Report Set",
        markers={
            # "Beat@K + short plan": "o",
            "Beat@K w/ detailed plan": "o",
            "Beat@K w/o gt plan": "o",
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
                xytext=(0, 8),
                ha="center",
                color="#3b3b3b",
                fontsize=8,
            )

    # Customize plot
    plt.tick_params(axis="both", direction="out", length=3, width=1)
    plt.xlabel("# Agent Rollouts (K)")
    plt.ylabel("% Problems")
    plt.xticks(k_values)
    plt.ylim(-1, 42)
    plt.grid(False)  #  linestyle="-", alpha=0.005)
    plt.legend(title=None)

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, "beat_at_k.backtranslate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")


if __name__ == "__main__":
    main()
