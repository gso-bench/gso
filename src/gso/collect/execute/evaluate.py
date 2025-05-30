import re
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

from gso.constants import EXPS_DIR
from gso.utils.io import load_problems


def has_non_python_changes(commit):
    """Check if commit includes changes to non-Python files."""
    ignore_list = ["py", "rst", "md", "txt", "yml", "toml", "gitignore"]
    return any(file.split(".")[-1] not in ignore_list for file in commit.files_changed)


def parse_times(time_str):
    pattern = r"Execution time:\s+([\d\.]+)s"
    times = []
    for line in time_str.strip().split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            seconds = float(match.group(1))
            times.append(seconds)
    return times


def compute_stats(times):
    if not times or len(times) == 0:
        return None, None

    mean = np.mean(times)
    std_dev = np.std(times, ddof=1)
    if np.isnan(mean):
        return None, None
    return mean, std_dev


def print_prob_summary(prob):
    print("-" * 50)
    print("Problem.", prob.pid)
    print("  commits:", prob.num_commits())
    print("  tests:", prob.num_tests())
    print("  results:", prob.num_results())
    print("  valid_commits:", prob.num_valid_commits())
    print("  valid_tests:", prob.num_valid_tests())
    print("-" * 50)


def speedup_summary(
    prob,
    speedup_threshold=2,
    speedup_mode="target",
    non_python_only=False,
    python_only=False,
):
    # get the last run, both (key, results)
    last_run_key, lrun_res = list(prob.results.items())[-1]
    stats = {}
    valid_commits, opt_commits = set(), set()

    for ct in lrun_res:
        test = ct["test_file"]
        commit = next((c for c in prob.commits if c.quick_hash() == ct["commit"]), None)

        if commit is None:
            continue

        if non_python_only and not has_non_python_changes(commit):
            continue
        elif python_only and has_non_python_changes(commit):
            continue

        base_result = ct["base_result"]
        base_times = parse_times(base_result)
        base_mean, base_std = compute_stats(base_times)

        if "commit_result" in ct:
            comm_result = ct["commit_result"]
            comm_times = parse_times(comm_result)
            comm_mean, comm_std = compute_stats(comm_times)
            if comm_mean is None:
                comm_mean, comm_std = base_mean, base_std
        else:
            comm_mean, comm_std = base_mean, base_std

        target_result = ct["target_result"]
        target_times = parse_times(target_result)
        target_mean, target_std = compute_stats(target_times)

        if speedup_mode == "target":
            before_mean, after_mean = base_mean, target_mean
        elif speedup_mode == "commit":
            before_mean, after_mean = base_mean, comm_mean

        opt_perc = ((before_mean - after_mean) / before_mean) * 100
        speedup_factor = before_mean / after_mean
        loc_changed = commit.stats.get("num_non_test_edited_lines", 0)
        # print(f"  {prob.pid} ({test}): {opt_perc:.2f}% | {speedup_factor:.2f}x")

        if before_mean > after_mean and opt_perc > speedup_threshold:

            # if speedup_mode == "commit" and target_mean > comm_mean:
            #     valid_commits.add(commit.quick_hash())
            #     continue

            ct_stats = {
                "pid": prob.pid,
                "api": prob.api,
                "commit": ct["commit"],
                "test_id": ct["test_id"],
                "base_mean": base_mean,
                "base_std": base_std,
                "commit_mean": comm_mean,
                "commit_std": comm_std if not np.isnan(comm_std) else 0,
                "target_mean": target_mean,
                "target_std": target_std,
                "opt_perc": opt_perc,
                "speedup_factor": speedup_factor,
                "loc_changed": loc_changed,
            }
            key = f"{prob.pid}-{test}"
            stats.update({key: ct_stats})
            valid_commits.add(commit.quick_hash())
            opt_commits.add(commit.quick_hash())
        else:
            valid_commits.add(commit.quick_hash())
    return stats, valid_commits, opt_commits


def create_analysis_dataframe(problems) -> pd.DataFrame:
    """Convert problems data into a pandas DataFrame for analysis."""
    rows = []
    for pid, prob_stats in problems.items():
        for key, stats in prob_stats.items():
            rows.append(
                {
                    "key": key,
                    "pid": stats["pid"],
                    "commit": stats["commit"],
                    "test_id": stats["test_id"],
                    "base_time": stats["base_mean"],
                    "target_time": stats["target_mean"],
                    "base_std": stats["base_std"],
                    "target_std": stats["target_std"],
                    "opt_perc": stats["opt_perc"],
                    "speedup_factor": stats["speedup_factor"],
                    "loc_changed": stats["loc_changed"],
                }
            )
    return pd.DataFrame(rows)


#########  Plotting Functions #########


def plot_speedup_distribution(df: pd.DataFrame, output_dir: str):
    """Create a distribution plot of speedups across all tests."""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="opt_perc", bins=10, kde=True)
    plt.axvline(
        df["opt_perc"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {df["opt_perc"].mean():.2f}%',
    )
    plt.title("Distribution of Performance Improvements")
    plt.xlabel("Opt (%)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"{output_dir}/speedup_distribution.png")
    plt.close()


def plot_top_improvements(df: pd.DataFrame, output_dir: str, top_n: int = 30):
    """Create a horizontal bar plot of top N improvements."""
    top_improvements = df.nlargest(top_n, "opt_perc")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_improvements, y="pid", x="opt_perc")
    plt.title(f"Top {top_n} Performance Improvements")
    plt.xlabel("Opt (%)")
    plt.ylabel("PID")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_improvements.png")
    plt.close()


def plot_execution_times_distribution(df: pd.DataFrame, output_dir: str):
    """Create boxplots comparing base and target execution times with log scale."""
    plt.figure(figsize=(10, 6))
    plot_data = pd.DataFrame(
        {"Base Time": df["base_time"], "Target Time": df["target_time"]}
    )

    # Create boxplot
    sns.boxplot(data=plot_data)
    plt.title("Distribution of Base vs Target Execution Times")
    plt.ylabel("Execution Time (seconds)")
    plt.yscale("log")  # Use log scale for y-axis
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/execution_times_distribution.png")
    plt.close()


def plot_top_pids_by_time(df: pd.DataFrame, output_dir: str, top_n: int = 20):
    top_pids = df.groupby("pid")["base_time"].max().reset_index()
    top_pids = top_pids.nlargest(top_n, "base_time")
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=top_pids, y="pid", x="base_time", palette="viridis", hue="pid"
    )

    for i, v in enumerate(top_pids["base_time"]):
        ax.text(v + 0.1, i, f"{v:.2f}s", va="center")

    plt.title(f"Top {top_n} PIDs by Base Execution Time)")
    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("PID")
    plt.tight_layout()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/top_time_pids.png")
    plt.close()

    return top_pids


#########  Performance Summary #########


def create_performance_summary(df: pd.DataFrame) -> dict:
    """Generate comprehensive performance statistics."""
    summary = {
        "total_tests": len(df),
        "mean_speedup": df["opt_perc"].mean(),
        "median_speedup": df["opt_perc"].median(),
        "std_speedup": df["opt_perc"].std(),
        "max_speedup": df["opt_perc"].max(),
        "min_speedup": df["opt_perc"].min(),
    }
    return summary


def main(
    exp_id: str,
    specific_api: str | None = None,
    speedup_threshold: int = 2,
    loc_threshold: int = None,
    speedup_mode: str = "target",
    top_k: int = 10,
    non_python_only: bool = False,
    python_only: bool = False,
    top_by: str = "opt",
):
    """Updated main function incorporating enhanced analysis."""
    exp_dir = EXPS_DIR / f"{exp_id}"
    output_dir = "plots"

    all_problems = load_problems(exp_dir / f"{exp_id}_results.json")
    if specific_api:
        problems = [p for p in all_problems if p.api == specific_api]
        if not problems:
            raise ValueError(f"No problem found for API: {specific_api}")
    else:
        problems = all_problems

    num_valid = 0
    opt_problems = {}
    err_problems = []
    opt_apis = set()
    all_commits = set()
    valid_commits_all = set()
    opt_commits_all = set()

    for prob in problems:
        all_commits.update(
            [c.quick_hash() for c in prob.commits if c.date.year >= 2016]
        )
        if prob.is_valid():
            stats, valid_commits, opt_commits = speedup_summary(
                prob,
                speedup_threshold=speedup_threshold,
                speedup_mode=speedup_mode,
                non_python_only=non_python_only,
                python_only=python_only,
            )
            valid_commits_all.update(valid_commits)
            opt_commits_all.update(opt_commits)

            num_valid += 1
            if stats:
                opt_problems[prob.pid] = stats
                for _, v in stats.items():
                    opt_apis.add(v["api"])
        else:
            err_problems.append(prob)

    if len(opt_problems) == 0:
        print("\n=== Performance Analysis Summary ===")
        print(f"Total problems: {len(problems)}")
        print(f"Valid problems: {num_valid} ({num_valid/len(problems)*100:.2f}%)")
        print("No optimization problems found!!")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    df = create_analysis_dataframe(opt_problems)
    plot_speedup_distribution(df, output_dir)
    plot_top_improvements(df, output_dir)
    plot_execution_times_distribution(df, output_dir)
    plot_top_pids_by_time(df, output_dir, top_n=20)
    summary = create_performance_summary(df)

    # Print summary report
    print("\n=== Performance Analysis Summary ===")
    print(f"Total problems: {len(problems)}")
    print(f"Valid problems: {num_valid} ({num_valid/len(problems)*100:.2f}%)")
    print(
        f"Optimized problems: {len(opt_problems)} ({len(opt_problems)/num_valid*100:.2f}%)"
    )
    print(f"Optimized APIs: {opt_apis}")
    print("\nErrored APIs:")
    for p in err_problems:
        commits = ", ".join(
            [f"{c.quick_hash()} ({c.date.strftime("%Y")})" for c in p.commits][:10]
        )
        print(f"  {p.api} : {commits}")

    print("\nTest Analysis:")
    print(f"  Total tests analyzed: {summary['total_tests']}")
    print(f"  Mean Opt: {summary['mean_speedup']:.2f}%")
    print(f"  Median Opt: {summary['median_speedup']:.2f}%")
    print(f"  Standard deviation: {summary['std_speedup']:.2f}%")
    print(f"  Max Opt: {summary['max_speedup']:.2f}%")
    print(f"  Min Opt: {summary['min_speedup']:.2f}%")
    print(
        f"\nSpeedup distribution:\n{df['opt_perc'].describe(percentiles=[0,0.05,0.1,0.2,0.4,0.5,0.6,0.8,0.9,0.95,1])}"
    )
    print("=" * 35)

    print("\nCommit Analysis:")
    print(f"  Total commits: {len(all_commits)}")
    print(f"  Total valid commits: {len(valid_commits_all)}")
    print(f"  Total optimized commits: {len(opt_commits_all)}")

    best = (
        df.groupby(["pid", "commit"])
        .agg({"opt_perc": "max", "speedup_factor": "max", "loc_changed": "first"})
        .reset_index()
    )

    if top_by == "opt":
        print("\nTop problems by Opt (best result per pid-commit):")
        for i, row in best.nlargest(top_k, "opt_perc").iterrows():
            print(
                f"  {row['pid']} ({row['commit']}): {row['opt_perc']:.2f}% | {row['speedup_factor']:.2f}x"
            )

    if top_by == "loc":
        print("\nTop problems by LoC (best result per pid-commit):")
        for i, row in best.nlargest(top_k, "loc_changed").iterrows():
            print(f"  {row['pid']} ({row['commit']}): {row['loc_changed']}")

    if loc_threshold is not None and top_by == "loc_opt":
        print(f"\nTop commits by Opt > {speedup_threshold}% & LoC > {loc_threshold}:")
        top = best[best["loc_changed"] > loc_threshold]
        top_commits = top.nlargest(top_k, "loc_changed")["commit"].unique()
        for commit in top_commits:
            max_loc = top[top["commit"] == commit]["loc_changed"].max()
            print(f"  {commit} ({max_loc} lines):")
            for _, row in top[top["commit"] == commit].iterrows():
                print(
                    f"      {row['pid']} ({row['opt_perc']:.2f}% | {row['speedup_factor']:.2f}x)"
                )

    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze performance results")
    parser.add_argument("-e", "--exp_id", type=str, help="Experiment ID", required=True)
    parser.add_argument("-a", "--api", type=str, help="Specific API", required=False)
    parser.add_argument(
        "-t", "--speedup_threshold", type=int, help="Speedup threshold", default=2
    )
    parser.add_argument(
        "-l", "--loc_threshold", type=int, help="Lines of code threshold", default=None
    )
    parser.add_argument(
        "-m",
        "--speedup_mode",
        type=str,
        help="Speedup mode (target or commit)",
        default="target",
    )
    parser.add_argument(
        "-k", "--top_k", type=int, help="Top results to show", default=10
    )
    parser.add_argument(
        "--non-python-only",
        action="store_true",
        help="Only use commits with non-Python code changes",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only use commits with Python code changes",
    )
    parser.add_argument("--top_by", choices=["loc", "opt", "loc_opt"], default="opt")
    args = parser.parse_args()

    if args.top_by == "loc_opt" and args.loc_threshold is None:
        raise ValueError("loc_threshold must be provided for top_by='loc_opt'")

    df, summary = main(
        args.exp_id,
        args.api,
        args.speedup_threshold,
        args.loc_threshold,
        args.speedup_mode,
        args.top_k,
        args.non_python_only,
        args.python_only,
        args.top_by,
    )
