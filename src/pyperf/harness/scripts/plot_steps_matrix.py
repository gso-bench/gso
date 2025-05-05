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

from pyperf.harness.utils import natural_sort_key
from pyperf.harness.scripts.helpers import *

# Add argument parsing
parser = argparse.ArgumentParser(description="Create a matrix plot for beat@K evaluations")
parser.add_argument("--model_name", type=str, help="Model name", required=True)
parser.add_argument("--eval_reports", type=str, nargs="+", help="evaluated reports", required=True)
parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
parser.add_argument("--k_values", type=int, nargs="+", default=[1, 2, 4, 8], help="K values to plot")
parser.add_argument("--l_values", type=int, nargs="+", default=[50, 100, 200, 400], help="Max iterations values to plot")
parser.add_argument("--num_trials", type=int, default=500, help="Number of bootstrap trials")
args = parser.parse_args()

# Create plots directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
args.l_values.sort(reverse=True)

# Function to extract maxiter (L) from filename
def extract_maxiter(filename):
    match = re.search(r'maxiter_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# Create a matrix to store the beat@k values
matrix_data = np.zeros((len(args.l_values), len(args.k_values)))
matrix_data.fill(np.nan)  # Fill with NaN initially

# Get all report files
reports = sorted(args.eval_reports, key=natural_sort_key)

# Process each L value and K value combination
for l_idx, l_value in enumerate(args.l_values):
    l_reports = [r for r in reports if extract_maxiter(r) == l_value]
    
    for k_idx, k_value in enumerate(args.k_values):
        if len(l_reports) >= k_value:
            _, _, commit_at_k_rates, _ = calculate_beat_at_k_smooth(
                l_reports[:k_value], k_value, fixed_first_run=False, num_trials=args.num_trials
            )
            matrix_data[l_idx, k_idx] = commit_at_k_rates[k_value-1][0]

# Create a DataFrame for the matrix plot
df_matrix = pd.DataFrame(
    matrix_data, 
    index=[f"{l}" for l in args.l_values],
    columns=[f"{k}" for k in args.k_values]
)
print(df_matrix)

# Create the plot
plt.figure(figsize=(6, 6))
setup_plot_style()

# Create a custom colormap that maps NaN to black
cmap = sns.color_palette("Blues_r", as_cmap=True)
cmap.set_bad('#303030')

# Create heatmap
ax = sns.heatmap(
    df_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap=cmap,
    cbar=False,  # Remove colorbar
    # linewidths=0.5,  # No gaps between cells
    # linecolor='#303030',  # Color of the lines between cells
)

# Add a black border around the heatmap
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('#303030')

# Customize plot
plt.xlabel("# Rollouts (K)")
plt.ylabel("# Steps (L)")
plt.tight_layout()

# Save the plot
output_path = os.path.join(args.output_dir, f"beat_matrix.{args.model_name}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Matrix plot saved as {output_path}")