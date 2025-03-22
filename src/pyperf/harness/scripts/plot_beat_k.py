import subprocess
import re
import matplotlib.pyplot as plt
import os
import json
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
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Store results for each k
k_values = []
base_values = []
commit_values = []
main_values = []

# sort report files
reports = sorted(args.eval_reports, key=natural_sort_key)[: args.k]

# raise if less than k reports are found
if len(reports) < args.k:
    raise ValueError(f"Found {len(reports)} reports, expected {args.k}")


for k in tqdm(range(1, args.k + 1), desc="Processing K predictions"):
    reports_at_k = reports[:k]

    # merge reports
    print(f"{'---'*4} beat@{k} {'---'*4}")
    merged_results = merge_reports(reports_at_k, k)
    print(f"\n{'---' * 10}\n\n")
    summary = merged_results["summary"]
    opt_base = summary["improved_over_base"] / summary["total_instances"]
    opt_commit = summary["improved_over_commit"] / summary["total_instances"]
    opt_main = summary["improved_over_main"] / summary["total_instances"]

    k_values.append(k)
    base_values.append(opt_base)
    commit_values.append(opt_commit)
    main_values.append(opt_main)


# convert y values to percentage
base_values = [x * 100 for x in base_values]
commit_values = [x * 100 for x in commit_values]
main_values = [x * 100 for x in main_values]

plt.figure(figsize=(8, 6))
plt.rcParams.update({"font.size": 14})
plt.xticks(k_values)
plt.plot(k_values, base_values, "o-", label="beat(base)@k", linewidth=2)
plt.plot(k_values, commit_values, "o-", label="beat(commit)@k", linewidth=2)
if PLOT_MAIN:
    plt.plot(k_values, main_values, "o-", label="beat(main)@k", linewidth=2)

plt.xlabel("# Agent Rollouts (K)")
plt.ylabel("% Problems")
plt.legend()
plt.grid(True, linestyle="-", alpha=0.05)

# Save the plot
output_path = os.path.join(args.output_dir, f"beat_at_k.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
