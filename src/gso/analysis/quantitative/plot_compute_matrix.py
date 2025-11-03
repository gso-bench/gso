import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
import re
from pathlib import Path
import argparse
import glob
import matplotlib.colors as mcolors

from gso.harness.utils import natural_sort_key
from gso.analysis.quantitative.helpers import *

# Define the L-K combinations we want to plot
LK_PAIRS = [
    (400, 1),
    (200, 1),
    (200, 2),
    (100, 1),
    (100, 2),
    (100, 4),
    (50, 1),
    (50, 2),
    (50, 4),
    (50, 8),
]

# Get unique L and K values while preserving order
L_VALUES, K_VALUES = [], []
for l, k in LK_PAIRS:
    if l not in L_VALUES:
        L_VALUES.append(l)
    if k not in K_VALUES:
        K_VALUES.append(k)

# Sort L values in descending order
L_VALUES.sort(reverse=True)

# Add argument parsing
parser = argparse.ArgumentParser(
    description="Create a matrix plot for Opt@K evaluations"
)
parser.add_argument("--model_name", type=str, help="Model name", required=True)
parser.add_argument(
    "--eval_reports", type=str, nargs="+", help="evaluated reports", required=True
)
parser.add_argument(
    "--output_dir", type=str, default="plots", help="Directory to save plots"
)
parser.add_argument(
    "--num_trials", type=int, default=500, help="Number of bootstrap trials"
)
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)


# Function to extract maxiter (L) from filename
def extract_maxiter(filename):
    match = re.search(r"maxiter_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


# Create a matrix to store the Opt@K values
matrix_data = np.zeros((len(L_VALUES), len(K_VALUES)))
matrix_data.fill(np.nan)  # Fill with NaN initially

# Get all report files
reports = sorted(args.eval_reports, key=natural_sort_key)

# Process each L value and K value combination
for l_idx, l_value in enumerate(L_VALUES):
    l_reports = [r for r in reports if extract_maxiter(r) == l_value]
    # print(f"\nProcessing L = {l_value} with reports:\n{"\n".join(l_reports)}")

    for k_idx, k_value in enumerate(K_VALUES):
        if (l_value, k_value) in LK_PAIRS and len(l_reports) >= k_value:
            _, _, commit_at_k_rates, _ = calculate_opt_at_k_smooth(
                l_reports, k_value, fixed_first_run=False, num_trials=args.num_trials
            )
            matrix_data[l_idx, k_idx] = commit_at_k_rates[k_value - 1][0]

# Create a DataFrame for the matrix plot
df_matrix = pd.DataFrame(
    matrix_data, index=[f"{l}" for l in L_VALUES], columns=[f"{k}" for k in K_VALUES]
)
print(df_matrix)

# Create the plot
plt.figure(figsize=(5, 3), dpi=300)
setup_plot_style()


# if claude in model_name, use Oranges_r colormap
# else use Blues_r colormap
if "claude" in args.model_name:
    cmap = sns.color_palette("Oranges_r", as_cmap=True)
else:
    cmap = sns.color_palette("Blues_r", as_cmap=True)

# Create a custom colormap that maps NaN to black
cmap.set_bad("#303030")

# Create heatmap
ax = sns.heatmap(
    df_matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    cbar=False,  # Remove colorbar
    annot_kws={"size": 14, "weight": "bold"},  # Annotation text properties
    # linewidths=0.5,  # No gaps between cells
    # linecolor='#303030',  # Color of the lines between cells
)

# Add a black border around the heatmap
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color("#303030")

# Customize plot
plt.xlabel("# Rollouts (K)", fontsize=16)
plt.ylabel("# Steps (L)", fontsize=16)
plt.tight_layout()

# rotate y-tick labels to be horizontal
plt.yticks(rotation=0)
title = "claude-3.6" if args.model_name == "claude" else "o4-mini"
plt.title(title, fontsize=16, fontweight="bold")

# Save the plot
output_path = os.path.join(args.output_dir, f"compute_matrix.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Matrix plot saved as {output_path}")
