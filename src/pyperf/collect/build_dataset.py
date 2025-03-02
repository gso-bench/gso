import pandas as pd
from dataclasses import asdict
from argparse import ArgumentParser
from datasets import Dataset

from pyperf.constants import EXPS_DIR, DATASET_DIR, MIN_PROB_SPEEDUP, MAX_TEST_COUNT
from pyperf.data.problem import Problem
from pyperf.data.dataset import PyPerfInstance
from pyperf.data.perf import PerformanceCommit
from pyperf.execute.evaluate import speedup_summary, create_analysis_dataframe
from pyperf.utils.io import str2bool, load_problems
from pyperf.collect.pids import TEST_PROBLEMS


def create_instance(prob: Problem, commit_hash: str, test_ids: list[int]):
    """Create a single dataset instance from a executed problem"""
    commit: PerformanceCommit = [
        c for c in prob.commits if c.quick_hash() == commit_hash
    ][0]

    test_samples = prob.get_tests(commit_hash, test_ids)

    return {
        "instance_id": (prob.repo.full_name + "-" + commit_hash).replace("/", "__"),
        "repo": prob.repo.full_name,
        "base_commit": commit.commit_hash + "^",
        "api": prob.api,
        "test_scripts": test_samples,
        "hints_text": commit.message,
        "setup_commands": prob.setup_commands,
        "install_commands": prob.install_commands,
        "created_at": commit.date.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_dataset(problems):
    print(f"Loaded problems: {len(problems)}")

    test_pid_commits = {pid: commit for pid, commit in TEST_PROBLEMS}
    problems = [p for p in problems if p.pid in test_pid_commits]
    print(f"Filtered problems: {len(problems)}")

    valid_problems = [p for p in problems if p.is_valid()]
    print(f"Valid problems: {len(valid_problems)}")

    opt_stats = {}
    for prob in valid_problems:
        stats, _, _ = speedup_summary(prob, speedup_threshold=2, speedup_mode="target")
        if stats:
            opt_stats[prob.pid] = stats

    # create dataframe and filter to test commits
    opt_problems_df = create_analysis_dataframe(opt_stats)
    mask = opt_problems_df.apply(
        lambda r: r["commit"] == test_pid_commits.get(r["pid"]), axis=1
    )
    opt_problems_df = opt_problems_df[mask]

    # Filter by minimum speedup and take top K tests per prob
    opt_problems_df = (
        opt_problems_df[opt_problems_df["speedup_factor"] >= MIN_PROB_SPEEDUP]
        .sort_values(["pid", "speedup_factor"], ascending=[True, False])
        .groupby("pid")
        .head(MAX_TEST_COUNT)
    )
    assert len(opt_problems_df["pid"].unique()) == len(test_pid_commits)

    avg_loc = opt_problems_df["loc_changed"].mean()
    avg_opt_perc = opt_problems_df["opt_perc"].mean()
    avg_speedup_factor = opt_problems_df["speedup_factor"].mean()
    test_dist = opt_problems_df.groupby("pid").size().describe()

    # Create dataset instances for selected (problem, commit, test)
    dataset = []
    for pid, grp in opt_problems_df.groupby("pid"):
        prob = [p for p in valid_problems if p.pid == pid][0]
        inst_dict = create_instance(
            prob, grp["commit"].iloc[0], grp["test_id"].tolist()
        )
        inst = PyPerfInstance(**inst_dict)
        dataset.append(inst)

    print("Created dataset!\n\n------ Dataset Summary ------")
    print(f"Size: {len(dataset)}")
    print(f"Avg LOC: {avg_loc:.2f}")
    print(f"Avg Opt%: {avg_opt_perc:.2f}%")
    print(f"Avg speedup: {avg_speedup_factor:.2f}X")
    print(f"Test dist: {test_dist}")

    return dataset


def main(exp_id, push_to_hf, hf_username):
    if exp_id is None:
        raise NotImplementedError("Building dataset for all exps not supported yet")

    exp_dir = EXPS_DIR / f"{exp_id}"
    problems = load_problems(exp_dir / f"{exp_id}_results.json")

    # Build dataset
    dataset = build_dataset(problems)

    # Save dataset to jsonl file
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_df = pd.DataFrame([asdict(inst) for inst in dataset])
    dataset_name = f"pyperf_{exp_id}" if exp_id else "pyperf"
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
    parser.add_argument("--exp_id", type=str, help="Experiment ID", default=None)
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
