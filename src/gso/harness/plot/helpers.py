import json
import numpy as np
import random
import matplotlib.pyplot as plt


def calculate_opt_at_k_smooth(report_paths, N, fixed_first_run=False, num_trials=500):
    """
    Calculate Opt@K rates with bootstrapping, closely following the reference implementation.
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
        passed_ids = set(report["instance_sets"].get("passed_ids", []))
        opt_base_ids = set(report["instance_sets"].get("opt_base_ids", []))
        opt_commit_ids = set(report["instance_sets"].get("opt_commit_ids", []))
        opt_main_ids = set(report["instance_sets"].get("opt_main_ids", []))

        # All instance IDs: (just take any that appear in any of the classes)
        all_instance_ids = set()
        for key in report["instance_sets"]:
            if key.endswith("_ids"):
                all_instance_ids.update(report["instance_sets"][key])

        for instance_id in all_instance_ids:
            if instance_id not in id_rankings_dict:
                id_rankings_dict[instance_id] = {}

            # Store (opt_base, opt_commit, opt_main)
            id_rankings_dict[instance_id][run_id] = (
                instance_id in passed_ids,
                instance_id in opt_base_ids,
                instance_id in opt_commit_ids,
                instance_id in opt_main_ids,
            )

        # assert len(id_rankings_dict) == 122, f"Expected 122 instances"

    total_instances = len(id_rankings_dict)
    passed_at_n_trials = np.zeros((num_trials, N))
    base_at_n_trials = np.zeros((num_trials, N))
    commit_at_n_trials = np.zeros((num_trials, N))
    main_at_n_trials = np.zeros((num_trials, N))

    # Run multiple trials
    for trial in range(num_trials):
        pass_at_n = np.zeros(N)
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

                # Check if passed in any
                if any(r[1][0] for r in n_rankings):
                    pass_at_n[idx] += 1

                # Check if opt base in any
                if any(r[1][1] for r in n_rankings):
                    base_at_n[idx] += 1

                # Check if opt commit in any
                if any(r[1][2] for r in n_rankings):
                    commit_at_n[idx] += 1

                # Check if opt main in any
                if any(r[1][3] for r in n_rankings):
                    main_at_n[idx] += 1

        # Store results for this trial
        passed_at_n_trials[trial] = pass_at_n
        base_at_n_trials[trial] = base_at_n
        commit_at_n_trials[trial] = commit_at_n
        main_at_n_trials[trial] = main_at_n

    # Calculate means and standard deviations
    passed_at_n_mean = np.mean(passed_at_n_trials, axis=0)
    passed_at_n_std = np.std(passed_at_n_trials, axis=0)
    base_at_n_mean = np.mean(base_at_n_trials, axis=0)
    base_at_n_std = np.std(base_at_n_trials, axis=0)
    commit_at_n_mean = np.mean(commit_at_n_trials, axis=0)
    commit_at_n_std = np.std(commit_at_n_trials, axis=0)
    main_at_n_mean = np.mean(main_at_n_trials, axis=0)
    main_at_n_std = np.std(main_at_n_trials, axis=0)

    # Convert to percentages
    passed_at_n_mean_pct = [x / total_instances * 100 for x in passed_at_n_mean]
    passed_at_n_std_pct = [x / total_instances * 100 for x in passed_at_n_std]
    base_at_n_mean_pct = [x / total_instances * 100 for x in base_at_n_mean]
    base_at_n_std_pct = [x / total_instances * 100 for x in base_at_n_std]
    commit_at_n_mean_pct = [x / total_instances * 100 for x in commit_at_n_mean]
    commit_at_n_std_pct = [x / total_instances * 100 for x in commit_at_n_std]
    main_at_n_mean_pct = [x / total_instances * 100 for x in main_at_n_mean]
    main_at_n_std_pct = [x / total_instances * 100 for x in main_at_n_std]

    # Format results for plotting
    passed_at_k_rates = list(zip(passed_at_n_mean_pct, passed_at_n_std_pct))
    base_at_k_rates = list(zip(base_at_n_mean_pct, base_at_n_std_pct))
    commit_at_k_rates = list(zip(commit_at_n_mean_pct, commit_at_n_std_pct))
    main_at_k_rates = list(zip(main_at_n_mean_pct, main_at_n_std_pct))

    return passed_at_k_rates, base_at_k_rates, commit_at_k_rates, main_at_k_rates


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 15,
            # "axes.titlesize": 14,
            # "axes.labelsize": 13,
            # "xtick.labelsize": 11,
            # "ytick.labelsize": 11,
            # "legend.fontsize": 11,
            # "figure.titlesize": 16,
        }
    )

    # Set cleaner grid style
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5

    # Set cleaner axes style
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1.0

    # Set figure DPI for better resolution
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


# OpenAI - blues, Anthropic - oranges, Google - purples
MODEL_COLOR_MAP = {
    "gpt-4o": "#0072B2",  # OpenAI - darker blue
    "o3-mini": "#56B4E9",  # OpenAI - medium blue
    "o4-mini": "#89C4F4",  # OpenAI - lighter blue
    "claude-3.6": "#E69F00",  # Anthropic - darker orange
    "claude-3.7": "#F5A742",  # Anthropic - lighter orange
    "claude-4": "#FFC062",  # Anthropic - lightest orange
    "gemini-2.5": "#9467BD",  # Google - purple
}

METRICS_COLOR_MAP = {
    "Passed Tests": "#b3b3b3",  # gray
    "Has Speedup over Base": "#396ab1",  # blue
    "Opt@K": "#da7c2f",  # orange
    "Opt@K w/ gt plan": "#3d9651",  # green
    "Opt@K w/o gt plan": "#396ab1",  # blue
    "Negative": "#a52a2a",  # red
    "Positive": "#3d9651",  # green
}
