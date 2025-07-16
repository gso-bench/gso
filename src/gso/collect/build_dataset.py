import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
from argparse import ArgumentParser
from datasets import Dataset

from gso.constants import *
from gso.data.problem import Problem
from gso.data.dataset import GSOInstance
from gso.data.perf import PerformanceCommit
from gso.collect.execute.evaluate import speedup_summary, create_analysis_dataframe
from gso.utils.io import load_problems
from gso.collect.pids import TEST_PROBLEMS, LONG_RUNNING_PROBLEMS
from gso.collect.utils import prepare_prob_script, prepare_install_commands


def create_instance(prob: Problem, commit_hash: str, test_ids: list[int]):
    """Create a single dataset instance from a executed problem"""
    commit: PerformanceCommit = [
        c for c in prob.commits if c.quick_hash() == commit_hash
    ][0]

    opt_commit = commit.commit_hash
    base_commit = prob.base_commit if prob.base_commit != "" else opt_commit + "^"

    test_samples = prob.get_tests(commit_hash, test_ids)
    install_commands = prepare_install_commands(prob.install_commands)
    prob_script = prepare_prob_script(test_samples)

    return {
        "instance_id": (prob.repo.full_name + "-" + commit_hash).replace("/", "__"),
        "repo": prob.repo.full_name,
        "base_commit": base_commit,
        "opt_commit": opt_commit,
        "api": prob.api,
        "prob_script": prob_script,
        "tests": test_samples,
        "hints_text": commit.message,
        "setup_commands": prob.setup_commands,
        "install_commands": install_commands,
        "created_at": commit.date.strftime("%Y-%m-%d %H:%M:%S"),
        "gt_commit_message": commit.message,
        "gt_diff": commit.diff_text,
    }


def build_dataset(problems, exp_id, debug=False):
    print(f"Loaded problems: {len(problems)}")

    test_problems_list = (
        TEST_PROBLEMS[exp_id]
        if exp_id
        else [item for sublist in TEST_PROBLEMS.values() for item in sublist]
    )

    # Create a set of tuples for efficient membership checking
    test_pid_commits_set = set(test_problems_list)
    test_pids = set(pid for pid, _ in test_problems_list)
    print("Filtered problems: ", len(test_pid_commits_set))

    # identify problems by pid and validity
    problems = [p for p in problems if p.pid in test_pids]
    valid_problems = [p for p in problems if p.is_valid()]

    opt_stats = {}
    for prob in valid_problems:
        stats, _, _ = speedup_summary(prob, speedup_threshold=2, speedup_mode="commit")
        if stats:
            opt_stats[prob.pid] = stats

    # create dataframe and filter to test commits
    analysis_df = create_analysis_dataframe(opt_stats)
    mask = analysis_df.apply(
        lambda r: (r["pid"], r["commit"]) in test_pid_commits_set, axis=1
    )
    analysis_df = analysis_df[mask]

    # Filter by minimum speedup and take top K tests per prob
    opt_problems_df = (
        analysis_df[analysis_df["speedup_factor"] >= MIN_PROB_SPEEDUP]
        .sort_values(["pid", "commit", "speedup_factor"], ascending=[True, True, False])
        .groupby(["pid", "commit"])
        .head(MAX_TEST_COUNT)
    )

    # heuristic: if a problem has < 5 tests, add some tests with lower speedup
    for (pid, commit), group in opt_problems_df.groupby(["pid", "commit"]):
        if len(group) < LOW_TEST_IDEAL_TEST_COUNT:
            additional_tests = (
                analysis_df[
                    (analysis_df["pid"] == pid)
                    & (analysis_df["commit"] == commit)
                    & (analysis_df["speedup_factor"] > LOW_TEST_FALLBACK_SPEEDUP)
                    & (analysis_df["speedup_factor"] < MIN_PROB_SPEEDUP)
                    & (~analysis_df["test_id"].isin(group["test_id"]))
                ]
                .sort_values("speedup_factor", ascending=False)
                .head(LOW_TEST_IDEAL_TEST_COUNT - len(group))
            )

            if not additional_tests.empty:
                opt_problems_df = pd.concat([opt_problems_df, additional_tests])

    # heuristic: use fewer tests for extremely long runtime problems
    for pid, commit, max_test_count in LONG_RUNNING_PROBLEMS:
        long_running_mask = (opt_problems_df["pid"] == pid) & (
            opt_problems_df["commit"] == commit
        )
        if any(long_running_mask):
            # print(f">>> long running problem: {pid} - {commit}")
            problem_indices = opt_problems_df.index[long_running_mask]
            sorted_indices = (
                opt_problems_df.loc[problem_indices]
                .sort_values("speedup_factor", ascending=False)
                .index[:max_test_count]
            )
            drop_indices = [idx for idx in problem_indices if idx not in sorted_indices]
            opt_problems_df = opt_problems_df.drop(drop_indices)

    unique_pid_commits = set(zip(opt_problems_df["pid"], opt_problems_df["commit"]))
    print(f"Found {len(unique_pid_commits)} / {len(test_pid_commits_set)} probs")

    if debug:
        missing_problems = [
            (pid, commit)
            for pid, commit in test_pid_commits_set
            if (pid, commit) not in unique_pid_commits
        ]
        if missing_problems:
            print(f"Missing problems: {len(missing_problems)}")
            for pid, commit in missing_problems:
                print(f"  - {pid} - {commit}")

    loc_dist = opt_problems_df["loc_changed"].describe()
    speedup_dist = opt_problems_df["speedup_factor"].describe()
    test_dist = (
        opt_problems_df.groupby(["pid", "commit"])
        .size()
        .describe(percentiles=[0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 0.9])
    )

    # Create dataset instances for selected (problem, commit, test)
    dataset = []
    for (pid, commit), grp in opt_problems_df.groupby(["pid", "commit"]):
        prob = [p for p in valid_problems if p.pid == pid][0]
        inst_dict = create_instance(prob, commit, grp["test_id"].tolist())
        inst = GSOInstance(**inst_dict)
        dataset.append(inst)

    dataset_df = pd.DataFrame([asdict(inst) for inst in dataset])

    pd.set_option("display.float_format", "{:.2f}".format)
    print("Created dataset!\n\n------ Dataset Summary ------")
    print(f"Size: {len(dataset)}")
    print(f"Avg LoC: {loc_dist['mean']:.2f}")
    print(f"Avg Speedup: {speedup_dist['mean']:.2f}X\n")

    print(f"LoC dist:\n{loc_dist}\n")
    print(f"Test dist:\n{test_dist}\n")
    print(f"Repo dist:\n{dataset_df['repo'].value_counts()}\n")

    # plot the test count distribution in /plots using sns.histplot
    plt.figure(figsize=(10, 6))
    plt.title("Test count distribution")
    plot_data = pd.DataFrame(
        opt_problems_df.groupby(["pid", "commit"]).size(), columns=["Test Count"]
    )
    sns.histplot(plot_data, x="Test Count", bins=range(1, 11), discrete=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "test_count_dist.png")

    # show the groups with less than 6 tests
    low_tests = opt_problems_df.groupby(["pid", "commit"]).filter(lambda x: len(x) < 4)
    if not low_tests.empty:
        print(
            f"Low test groups (Count: {len(low_tests.drop_duplicates(subset=["pid", "commit"]))})"
        )
        low_tests_df = pd.DataFrame(
            low_tests.groupby(["pid", "commit"]).size(), columns=["Test Count"]
        ).reset_index()
        print(low_tests_df)

    return dataset


def main(exp_id, push_to_hf, hf_username, dataset_name=None, debug=False):
    exp_ids = TEST_PROBLEMS.keys() if exp_id is None else [exp_id]
    problems = [
        problem
        for eid in exp_ids
        for problem in load_problems((EXPS_DIR / f"{eid}" / f"{eid}_results.json"))
    ]

    if exp_id and exp_id not in TEST_PROBLEMS.keys():
        exp_id = None  # unset

    # Build dataset
    dataset = build_dataset(problems, exp_id, debug=debug)

    # Save dataset to jsonl file
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_df = pd.DataFrame([asdict(inst) for inst in dataset])
    if not dataset_name:
        dataset_name = f"gso_{exp_id}" if exp_id else "gso"
    dataset_df.to_json(
        DATASET_DIR / f"{dataset_name}_dataset.jsonl", orient="records", lines=True
    )

    if push_to_hf:
        hf_dataset = Dataset.from_pandas(dataset_df)
        hf_dataset.push_to_hub(
            f"{hf_username}/{dataset_name}", split="test", private=True
        )

    if debug:
        log_dir = Path("logs/tasks")
        log_dir.mkdir(parents=True, exist_ok=True)
        for inst in dataset:
            inst_dir = log_dir / inst.instance_id
            inst_dir.mkdir(parents=True, exist_ok=True)
            for i, test in enumerate(inst.tests):
                with open(inst_dir / f"gso_test_{i}.py", "w") as f:
                    f.write(test)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze performance results")
    parser.add_argument("--exp_id", type=str, help="Experiment ID", default=None)
    parser.add_argument("--dataset_name", type=str, help="Dataset name", default=None)
    parser.add_argument(
        "--push_to_hf",
        action="store_true",
        help="Push a HuggingFace dataset to hub",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        help="HuggingFace username",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()
    main(**vars(args))
