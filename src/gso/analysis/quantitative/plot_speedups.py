import os
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
from gso.harness.grading.metrics import speedup
from gso.analysis.quantitative.helpers import *

# Add argument parsing
parser = argparse.ArgumentParser(
    description="Plot speedup metrics from evaluation report"
)
parser.add_argument(
    "--eval_report", type=str, required=True, help="Path to evaluation report JSON file"
)
parser.add_argument("--model_name", type=str, help="Model name", required=True)
parser.add_argument(
    "--show_all_probs",
    action="store_true",
    default=False,
    help="Show all problems (default: show only problems where model outperforms commit)",
)
parser.add_argument(
    "--output_dir", type=str, default="plots", help="Directory to save plots"
)
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Load the report
report = json.load(open(os.path.expanduser(args.eval_report)))
opt_stats = report["opt_stats"]
instance_sets = report["instance_sets"]


def geomean_speedup(before_test_means, after_test_means):
    before_mean = np.mean(before_test_means)
    after_mean = np.mean(after_test_means)
    _, speedup_gm, speedup_gsd, speedup_hm, _ = speedup(
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

# Filter to instances where model outperforms commit unless show_all_probs is enabled
if not args.show_all_probs and "opt_commit_ids" in instance_sets:
    improved_instances = instance_sets["opt_commit_ids"]
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
speedups_main = [s if s else 0 for s in speedups_main]

max_speedup = max(speedups_model + speedups_commit + speedups_main)

# Clean up problem names for display
instances = [p.split("__")[1] for p in instances]  # Remove org name

# Create the plot
fig, ax = plt.subplots(figsize=(4, 3))
x = np.arange(len(instances))
width = 0.2

# Create bars
bars1 = ax.bar(x - width, speedups_model, width, label="Model Patch")
# bars2 = ax.bar(x, speedups_commit, width, label="Human Commit")
# bars3 = ax.bar(x + width, speedups_main, width, label="Main")

# Add labels
# add_top_labels(bars1)
# add_top_labels(bars2)
# add_top_labels(bars3)

# Apply log scale to y-axis
ax.set_yscale("log")
ax.set_ylabel(f"Speedup (Log Scale)")
ax.set_xlabel("Problem ID")
ax.set_ylim(top=max_speedup * 4)
ax.set_xticks(x)
ax.set_xticklabels(instances, rotation=30, ha="center", fontsize=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
setup_plot_style()
# Save the plot
output_path = os.path.join(args.output_dir, f"opt_per_prob.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
