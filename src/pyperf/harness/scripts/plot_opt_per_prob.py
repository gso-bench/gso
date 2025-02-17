import os
import matplotlib.pyplot as plt
import json
import numpy as np

from pyperf.harness.build_dataset import *
from pyperf.utils.io import load_problems, load_pyperf_dataset
from pyperf.constants import EXPS_DIR


def get_exec_results_helper(problems, speedup_mode="target"):
    valid_problems = [p for p in problems if p.is_valid()]
    opt_stats = {}
    for prob in valid_problems:
        stats, _, _ = speedup_summary(
            prob, speedup_threshold=2, speedup_mode=speedup_mode
        )
        if stats:
            opt_stats[prob.pid] = stats

    opt_problems_df = create_analysis_dataframe(opt_stats)
    opt_problems_df = get_most_optimized_commit_test_pairs(opt_problems_df)
    opt_problems_df = opt_problems_df[["key", "pid", "speedup_factor"]]
    return opt_problems_df


# Add value labels on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label non-zero bars
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


os.makedirs("plots", exist_ok=True)
exp_id = "pandas"
dataset_name = "pyperf_pandas_dataset.jsonl"


exp_dir = EXPS_DIR / f"{exp_id}"
problems = load_problems(exp_dir / f"{exp_id}_results.json")
dataset = load_pyperf_dataset(dataset_name, "test")
dataset_apis = [
    f"{exp_id}-{i.api.lower()}" for i in dataset if i.api is not None and i.api != ""
]
problems = [p for p in problems if p.pid in dataset_apis]


eval_report = "~/pyperf/claude.opt@10.test.report.json"
opt_stats = json.load(open(os.path.expanduser(eval_report)))["opt_stats"]

# Extract problem names and speedups
instances = list(opt_stats.keys())
speedups_base = [
    opt_stats[p]["speedup_base"] if opt_stats[p]["speedup_base"] is not None else 0
    for p in instances
]

commit_opt_df = get_exec_results_helper(problems, speedup_mode="commit")
main_opt_df = get_exec_results_helper(problems, speedup_mode="target")
speedups_main, speedups_commit = [], []

for inst in instances:
    commit = inst.split("-")[-1]  # get ccca5df from pandas-dev__pandas-ccca5df
    matches = main_opt_df[main_opt_df["key"].str.contains(commit)]
    speedup = matches["speedup_factor"].iloc[0] if not matches.empty else 0
    speedups_main.append(float(speedup))

    matches = commit_opt_df[commit_opt_df["key"].str.contains(commit)]
    speedup = matches["speedup_factor"].iloc[0] if not matches.empty else 0
    speedups_commit.append(float(speedup))


# Clean up problem names for display
instances = [p.split("__")[1] for p in instances]  # Remove org name
instances = [p[:12] for p in instances]  # Truncate for readability


plt.figure(figsize=(10, 6))
x = np.arange(len(instances))
width = 0.1  # Width of bars
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars1 = ax.bar(x - width, speedups_base, width, label="Base", color="#1f77b4")
bars2 = ax.bar(x, speedups_commit, width, label="Commit", color="#ff7f0e")
bars3 = ax.bar(x + width, speedups_main, width, label="Main", color="#2ca02c")
autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# Customize plot
ax.set_ylabel("Speedup Factor")
ax.set_title("Speedup across Commits")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.legend()
plt.tight_layout()
plt.savefig("plots/opt_per_prob.png", dpi=300, bbox_inches="tight")
plt.close()
