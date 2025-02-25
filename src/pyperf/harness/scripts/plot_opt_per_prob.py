import os
import matplotlib.pyplot as plt
import json
import numpy as np

from pyperf.harness.build_dataset import *

speedup_mode = "Factor"
# eval_report = "~/pyperf/reports/opt_k_reports/claude.opt@10.test.report.json"
eval_report = "~/pyperf/reports/opt_k_reports/o3-mini-high.opt@25.test.report.json"
opt_stats = json.load(open(os.path.expanduser(eval_report)))["opt_stats"]


def speedup(before_mean, after_mean, mode) -> float:
    if mode == "Factor":
        return before_mean / after_mean
    return ((before_mean - after_mean) / before_mean) * 100


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

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(instances))
width = 0.2  # Wider bars for better visibility

# Create bars
bars1 = ax.bar(x - width, speedups_model, width, label="Model", color="#1f77b4")
bars2 = ax.bar(x, speedups_commit, width, label="Commit", color="#ff7f0e")
bars3 = ax.bar(x + width, speedups_main, width, label="Main", color="#2ca02c")


# Function to add vertical labels
def add_vertical_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label non-zero bars
            # For very small values, don't show label
            if height < 1:
                continue

            # Format label differently based on magnitude
            if height >= 1000:
                label = f"{height:.2f}"
            elif height >= 100:
                label = f"{height:.2f}"
            else:
                label = f"{height:.2f}"

            # Position text in the middle of the bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,  # Position text in the middle of the bar
                label,
                ha="center",
                va="center",
                rotation="vertical",
                color="white",
                fontsize=9,
            )


# Add vertical labels inside bars
add_vertical_labels(bars1)
add_vertical_labels(bars2)
add_vertical_labels(bars3)

# Customize plot
ax.set_ylabel(f"Speedup ({speedup_mode})")
ax.set_title("Speedup achieved per problem", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.tick_params(axis="x", rotation=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right")

# Adjust y-axis scale
y_max = max(max(speedups_model), max(speedups_commit), max(speedups_main))
ax.set_ylim(0, y_max * 1.1)  # Add 10% padding at the top

# Add grid for better readability
ax.yaxis.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("plots/improved_speedup_chart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Improved plot saved as plots/improved_speedup_chart.png")

# Alternative version with log scale (helps when values have large differences)
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars
bars1 = ax.bar(x - width, speedups_model, width, label="Model", color="#1f77b4")
bars2 = ax.bar(x, speedups_commit, width, label="Commit", color="#ff7f0e")
bars3 = ax.bar(x + width, speedups_main, width, label="Main", color="#2ca02c")


# Add standard horizontal labels above bars
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


# Add labels
add_top_labels(bars1)
add_top_labels(bars2)
add_top_labels(bars3)

# Apply log scale to y-axis
ax.set_yscale("log")
ax.set_title("Speedup achieved per problem")
ax.set_ylabel(f"Speedup ({speedup_mode}) - Log Scale")
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("plots/opt_per_prob.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plot saved as plots/opt_per_prob.png")
