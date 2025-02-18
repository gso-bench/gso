import os
import matplotlib.pyplot as plt
import json
import numpy as np

from pyperf.harness.build_dataset import *
from pyperf.utils.io import load_problems, load_pyperf_dataset
from pyperf.constants import EXPS_DIR

speedup_mode = "Factor"
# speedup_mode = "Percentage"

os.makedirs("plots", exist_ok=True)
eval_report = "~/pyperf/claude.opt@10.test.report.json"
opt_stats = json.load(open(os.path.expanduser(eval_report)))["opt_stats"]


def speedup(before_mean, after_mean, mode) -> float:
    if mode == "Factor":
        return before_mean / after_mean
    return ((before_mean - after_mean) / before_mean) * 100


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


# Extract speedups
instances = list(opt_stats.keys())
speedups_model = [
    speedup(opt_stats[p]["base_mean"], opt_stats[p]["patch_mean"], speedup_mode)
    for p in instances
]

speedups_commit = [
    speedup(opt_stats[p]["base_mean"], opt_stats[p]["commit_mean"], speedup_mode)
    for p in instances
]

speedups_main = [
    speedup(opt_stats[p]["base_mean"], opt_stats[p]["main_mean"], speedup_mode)
    for p in instances
]

# Clean up problem names for display
instances = [p.split("__")[1] for p in instances]  # Remove org name
instances = [p[:12] for p in instances]  # Truncate for readability


plt.figure(figsize=(10, 6))
x = np.arange(len(instances))
width = 0.1  # Width of bars
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars1 = ax.bar(x - width, speedups_model, width, label="Model", color="#1f77b4")
bars2 = ax.bar(x, speedups_commit, width, label="Commit", color="#ff7f0e")
bars3 = ax.bar(x + width, speedups_main, width, label="Main", color="#2ca02c")
autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# Customize plot
ax.set_ylabel(f"Speedup ({speedup_mode})")
ax.set_title("Speedup achieved per problem")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.legend()
plt.tight_layout()
plt.savefig("plots/opt_per_prob.png", dpi=300, bbox_inches="tight")
plt.close()
