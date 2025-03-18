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

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, base_values, "o-", label="beat(base)@k", linewidth=2)
plt.plot(k_values, commit_values, "s-", label="beat(commit)@k", linewidth=2)
plt.plot(k_values, main_values, "^-", label="beat(main)@k", linewidth=2)

plt.xlabel("# Agent Rollouts (K)")
plt.ylabel("% Problems")
plt.title("Inference time scaling for beat@K")
plt.legend()
plt.grid(True)

# Save the plot
output_path = os.path.join(args.output_dir, f"beat_at_k.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
