import os
import matplotlib.pyplot as plt
import json
import numpy as np
from pyperf.harness.grading.metrics import speedup

filter_improved_commit = True  # Set to False to show all problems

# eval_report = "~/pyperf/reports/opt_k_reports/claude.opt@10.test.report.json"
eval_report = "~/pyperf/reports/opt_k_reports/o3-mini.opt@10.test.report.json"
report = json.load(open(os.path.expanduser(eval_report)))
opt_stats = report["opt_stats"]
instance_sets = report["instance_sets"]


def geomean_speedup(before_test_means, after_test_means):
    before_mean = np.mean(before_test_means)
    after_mean = np.mean(after_test_means)
    _, _, speedup_gm = speedup(
        before_mean, after_mean, before_test_means, after_test_means
    )
    return speedup_gm


def add_top_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label non-zero bars
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )


# Extract speedups
instances = list(opt_stats.keys())

# Filter to instances where model outperforms commit if flag is enabled
if filter_improved_commit and "improved_commit_ids" in instance_sets:
    improved_instances = instance_sets["improved_commit_ids"]
    instances = [instance for instance in instances if instance in improved_instances]


speedups_model = [
    geomean_speedup(
        opt_stats[p]["per_test_means"]["base"],
        opt_stats[p]["per_test_means"]["patch"],
    )
    for p in instances
]

speedups_commit = [
    geomean_speedup(
        opt_stats[p]["per_test_means"]["base"],
        opt_stats[p]["per_test_means"]["commit"],
    )
    for p in instances
]

speedups_main = [
    geomean_speedup(
        opt_stats[p]["per_test_means"]["base"],
        opt_stats[p]["per_test_means"]["main"],
    )
    for p in instances
]

max_speedup = max(speedups_model + speedups_commit + speedups_main)

# Clean up problem names for display
instances = [p.split("__")[1] for p in instances]  # Remove org name

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(instances))
width = 0.2

# Create bars
bars1 = ax.bar(x - width, speedups_model, width, label="Model", color="#1f77b4")
bars2 = ax.bar(x, speedups_commit, width, label="Commit", color="#ff7f0e")
bars3 = ax.bar(x + width, speedups_main, width, label="Main", color="#2ca02c")

# Add labels
add_top_labels(bars1)
add_top_labels(bars2)
add_top_labels(bars3)

# Apply log scale to y-axis
ax.set_yscale("log")
ax.set_title("Speedup achieved per problem")
ax.set_ylabel(f"Speedup Factor - Log Scale")
ax.set_ylim(top=max_speedup * 4)
ax.set_xticks(x)
ax.set_xticklabels(instances, rotation=30, ha="center", fontsize=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("plots/opt_per_prob.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as plots/opt_per_prob.png")
