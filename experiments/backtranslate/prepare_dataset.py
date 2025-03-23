import pandas as pd
from dataclasses import asdict
from argparse import ArgumentParser
from datasets import Dataset

from pyperf.constants import EXPS_DIR, DATASET_DIR, MIN_PROB_SPEEDUP, MAX_TEST_COUNT
from pyperf.data.problem import Problem
from pyperf.data.dataset import PyPerfInstance
from pyperf.data.perf import PerformanceCommit
from pyperf.execute.evaluate import speedup_summary, create_analysis_dataframe
from pyperf.utils.io import load_problems
from pyperf.collect.pids import TEST_PROBLEMS
from pyperf.collect.utils import prepare_prob_script


def create_instance(prob: Problem, commit_hash: str, test_ids: list[int]):
    """Create a single dataset instance from a executed problem"""
    commit: PerformanceCommit = [
        c for c in prob.commits if c.quick_hash() == commit_hash
    ][0]

    test_samples = prob.get_tests(commit_hash, test_ids)
    prob_script = prepare_prob_script(test_samples)

    return {
        "instance_id": (prob.repo.full_name + "-" + commit_hash).replace("/", "__"),
        "repo": prob.repo.full_name,
        "base_commit": commit.commit_hash + "^",
        "api": prob.api,
        "prob_script": prob_script,
        "tests": test_samples,
        "hints_text": commit.message,
        "setup_commands": prob.setup_commands,
        "install_commands": prob.install_commands,
        "created_at": commit.date.strftime("%Y-%m-%d %H:%M:%S"),
        "gt_diff": commit.diff_text,
        "files_changed": commit.files_changed,
        "functions_changed": commit.functions_changed,
        "commit_stats": commit.stats,
    }


def build_dataset(problems, exp_id):
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
    opt_problems_df = create_analysis_dataframe(opt_stats)
    mask = opt_problems_df.apply(
        lambda r: (r["pid"], r["commit"]) in test_pid_commits_set, axis=1
    )
    opt_problems_df = opt_problems_df[mask]

    # Filter by minimum speedup and take top K tests per prob
    opt_problems_df = (
        opt_problems_df[opt_problems_df["speedup_factor"] >= MIN_PROB_SPEEDUP]
        .sort_values(["pid", "commit", "speedup_factor"], ascending=[True, True, False])
        .groupby(["pid", "commit"])
        .head(MAX_TEST_COUNT)
    )
    unique_pid_commits = set(zip(opt_problems_df["pid"], opt_problems_df["commit"]))
    print(f"Found {len(unique_pid_commits)} / {len(test_pid_commits_set)} probs")

    loc_dist = opt_problems_df["loc_changed"].describe()
    speedup_dist = opt_problems_df["speedup_factor"].describe()
    test_dist = opt_problems_df.groupby(["pid", "commit"]).size().describe()

    # Create dataset instances for selected (problem, commit, test)
    dataset = []
    for (pid, commit), grp in opt_problems_df.groupby(["pid", "commit"]):
        prob = [p for p in valid_problems if p.pid == pid][0]
        inst_dict = create_instance(prob, commit, grp["test_id"].tolist())
        dataset.append(inst_dict)

    pd.set_option("display.float_format", "{:.2f}".format)
    print("Created dataset!\n\n------ Dataset Summary ------")
    print(f"Size: {len(dataset)}")
    print(f"Avg LoC: {loc_dist['mean']:.2f}")
    print(f"Avg Speedup: {speedup_dist['mean']:.2f}X\n")

    print(f"LoC dist:\n{loc_dist}\n")
    print(f"Speedup dist:\n{speedup_dist}\n")
    print(f"Test dist:\n{test_dist}")

    return dataset


def main(push_to_hf, hf_username, dataset_name):
    exp_ids = TEST_PROBLEMS.keys()
    problems = [
        problem
        for eid in exp_ids
        for problem in load_problems((EXPS_DIR / f"{eid}" / f"{eid}_results.json"))
    ]

    # Build dataset
    dataset = build_dataset(problems, exp_id=None)

    # Save dataset to jsonl file
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_df = pd.DataFrame([inst for inst in dataset])
    dataset_df.to_json(
        DATASET_DIR / f"{dataset_name}_dataset.jsonl", orient="records", lines=True
    )

    if push_to_hf:
        hf_dataset = Dataset.from_pandas(dataset_df)
        hf_dataset.push_to_hub(
            f"{hf_username}/{dataset_name}", split="test", private=True
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze performance results")
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

    args = parser.parse_args()
    main(**vars(args))
