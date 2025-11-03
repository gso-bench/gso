import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import glob
from tqdm import tqdm

from gso.harness.utils import natural_sort_key
from gso.analysis.quantitative.helpers import *

# =============================================================================
# CONFIGURATION - Just configure what you want to plot
# =============================================================================

MODEL_CONFIGS = {
    "o4-mini": "~/gso/reports/archives/scale/o4-mini_maxiter_100_N_v0.35.0-no-hint-run_*scale*",
    "claude-3.6": "~/gso/reports/archives/scale/claude-3-5-sonnet-v2-20241022_maxiter_100_N_v0.35.0-no-hint-run_*scale*",
}

METRICS_TO_PLOT = ["passed", "speedup", "commit"]
K_MAX = 10
OUTPUT_DIR = "plots"
FIXED_FIRST_RUN = False
NUM_TRIALS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)
is_single_model = len(MODEL_CONFIGS) == 1
is_multi_model = len(MODEL_CONFIGS) > 1

# Process each model
all_data = []

for model_name, pattern in MODEL_CONFIGS.items():
    reports = sorted(glob.glob(os.path.expanduser(pattern)), key=natural_sort_key)
    if not reports or len(reports) < K_MAX:
        print(f"Skipping {model_name}: {len(reports) if reports else 0} reports")
        continue

    print(f"Processing {model_name} with {len(reports)} reports...")
    passed_at_k_rates, base_at_k_rates, commit_at_k_rates, _ = (
        calculate_opt_at_k_smooth(reports, K_MAX, FIXED_FIRST_RUN, NUM_TRIALS)
    )

    if is_single_model and METRICS_TO_PLOT:
        # Single model with multiple metrics
        metric_data = {
            "passed": (passed_at_k_rates, "Passed Tests"),
            "speedup": (base_at_k_rates, "Has Speedup over Base"),
            "commit": (commit_at_k_rates, "Opt@K"),
        }

        for metric in METRICS_TO_PLOT:
            rates, metric_name = metric_data[metric]
            for k, rate_info in enumerate(rates, 1):
                error = rate_info[1] if k < K_MAX else 0
                all_data.append(
                    {
                        "k": k,
                        "Rate": rate_info[0],
                        "Error": error,
                        "Metric": metric_name,
                        "Lower": rate_info[0] - error,
                        "Upper": rate_info[0] + error,
                    }
                )
    else:
        # Multi model with Opt@K only
        for k, rate_info in enumerate(commit_at_k_rates, 1):
            error = rate_info[1] if k < K_MAX else 0
            all_data.append(
                {
                    "k": k,
                    "Rate": rate_info[0],
                    "Error": error,
                    "Model": model_name,
                    "Lower": rate_info[0] - error,
                    "Upper": rate_info[0] + error,
                }
            )

if not all_data:
    print("No data to plot!")
    exit(1)

df = pd.DataFrame(all_data)
plt.figure(figsize=(8, 6) if is_single_model else (6, 4))
setup_plot_style()

if is_single_model:
    hue_col = "Metric"
    style_col = "Metric"
    palette = METRICS_COLOR_MAP
    markers = {"Opt@K": "o", "Has Speedup over Base": "o", "Passed Tests": "o"}
    ylabel = "% Problems"
    ylim = (-3, 101)
    legend_loc = "lower right"
    output_path = os.path.join(
        OUTPUT_DIR, f"opt_at_k.{list(MODEL_CONFIGS.keys())[0]}.png"
    )
else:
    hue_col = "Model"
    style_col = "Model"
    palette = {
        model: MODEL_COLOR_MAP.get(model, "#999999") for model in df["Model"].unique()
    }
    markers = True
    ylabel = "Opt@K (%)"
    ylim = (0, 25)
    legend_loc = "upper left"
    output_path = os.path.join(OUTPUT_DIR, "opt_at_k_comparison_new.png")

sns.lineplot(
    data=df,
    x="k",
    y="Rate",
    hue=hue_col,
    style=style_col,
    markers=markers,
    markeredgewidth=0,
    markersize=5,
    dashes=False,
    palette=palette,
)

# Error bands
for group in df[hue_col].unique():
    group_data = df[df[hue_col] == group]
    color = (
        palette[group] if isinstance(palette, dict) else palette.get(group, "#999999")
    )
    plt.fill_between(
        group_data["k"],
        group_data["Lower"],
        group_data["Upper"],
        alpha=0.2,
        color=color,
        linewidth=0,
    )

# Value labels
if is_single_model:
    # Single model: annotate specific indices
    for metric, color in METRICS_COLOR_MAP.items():
        metric_data = df[df["Metric"] == metric]
        for idx, (_, row) in enumerate(metric_data.iterrows()):
            if idx in [0, 2, 4, 7, 9]:
                plt.annotate(
                    f'{row["Rate"]:.1f}',
                    (row["k"], row["Rate"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    color="#3b3b3b",
                    fontsize=12,
                )
else:
    # Multi model: smart positioning based on relative performance
    rates_by_k = df.pivot(index="k", columns="Model", values="Rate")
    annotation_ks = [2, 4, 6, 8, 10]

    for model in df["Model"].unique():
        model_data = df[df["Model"] == model].sort_values("k")
        for _, row in model_data.iterrows():
            k = row["k"]
            if k not in annotation_ks:
                continue

            y = row["Rate"]
            other = [m for m in df["Model"].unique() if m != model][0]
            other_y = rates_by_k.loc[k, other]

            offset = 8
            xytext = (0, offset) if y > other_y else (0, -offset)
            va = "bottom" if y > other_y else "top"

            plt.annotate(
                f"{y:.1f}",
                (k, y),
                textcoords="offset points",
                xytext=xytext,
                ha="center",
                va=va,
                color="#3b3b3b",
                fontsize=12,
            )

plt.ylabel(ylabel)
plt.ylim(ylim)
plt.legend(title=None, loc=legend_loc)
plt.tick_params(axis="both", direction="out", length=3, width=1)
plt.xlabel("# Rollouts (K)")
plt.xticks(list(range(1, K_MAX + 1)))
plt.grid(False)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")
