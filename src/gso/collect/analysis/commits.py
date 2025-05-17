import os
import re
import json
import argparse
from pathlib import Path

from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool

from r2e.llms.llm_args import LLMArgs
from r2e.llms.completions import LLMCompletions

from gso.data import PerformanceCommit, PerfAnalysis
from gso.collect.analysis.parser import CommitParser
from gso.collect.analysis.retriever import Retriever
from gso.collect.analysis.prompt import *
from gso.collect.analysis.utils import *
from gso.constants import *
from gso.utils.io import *

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")
MAX_COMMIT_TOKENS = 20000
MAX_OAI_TOKENS = 90000
THRESHOLD = 200
SKIP_API_ANALYSIS = False  # Set to True to skip API analysis


class PerfCommitAnalyzer:
    @staticmethod
    def parse_diff_for_stats(
        commit: PerformanceCommit, repo_path: Path
    ) -> dict[str, int]:
        parser = CommitParser()
        diff = parser.parse_commit(
            commit.old_commit_hash,
            commit.commit_hash,
            commit.diff_text,
            commit.message,
            commit.date,
            repo_path,
        )

        stats = {
            "num_test_files": diff.num_test_files,
            "num_non_test_files": diff.num_non_test_files,
            "only_test_files": diff.num_files == diff.num_test_files,
            "only_non_test_files": diff.num_files == diff.num_non_test_files,
            "num_files": diff.num_files,
            "num_hunks": diff.num_hunks,
            "num_edited_lines": diff.num_edited_lines,
            "num_non_test_edited_lines": diff.num_non_test_edited_lines,
            "commit_year": diff.commit_date.year,
        }

        return stats

    @staticmethod
    def process_commit(
        commit_hash: str, repo_path: Path, max_year: int
    ) -> PerformanceCommit:
        # commit subject
        subject = run_git_command(
            ["git", "show", "--no-patch", "--format=%s", commit_hash], cwd=repo_path
        )

        # commit message
        message = run_git_command(
            ["git", "show", "--no-patch", "--format=%B", commit_hash], cwd=repo_path
        )

        # commit date
        date_str = run_git_command(
            ["git", "show", "-s", "--format=%cd", commit_hash], cwd=repo_path
        )
        date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")

        if max_year and date.year <= max_year:
            return None

        # changed files
        files_changed = run_git_command(
            ["git", "show", "--name-only", "--format=", commit_hash], cwd=repo_path
        ).split("\n")

        # commit diff
        old_commit_hash = f"{commit_hash}^"

        try:
            diff_text = run_git_command(
                ["git", "diff", "-p", old_commit_hash, commit_hash], cwd=repo_path
            )
        except:
            return None  # mostly for the root commit

        return PerformanceCommit(
            commit_hash=commit_hash,
            subject=subject,
            message=message,
            date=date,
            files_changed=files_changed,
            diff_text=diff_text,
            repo_path=repo_path,
        )

    ######################### LLM-based Commit Filtering #########################

    @staticmethod
    def analysis_prompt(commit: PerformanceCommit):
        prompt = PERF_ANALYSIS_MESSAGE.format(
            diff_text=commit.diff_text, message=commit.message
        )

        if count_tokens(prompt) > MAX_COMMIT_TOKENS:
            diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
            prompt = PERF_ANALYSIS_MESSAGE.format(
                diff_text=diff_text, message=commit.message
            )

        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    @staticmethod
    def llm_analysis(
        commits: list[PerformanceCommit], repo_path: Path, verbose: bool = False
    ):
        prompts = [PerfCommitAnalyzer.analysis_prompt(commit) for commit in commits]

        args = LLMArgs(
            model_name="o3-mini",
            cache_batch_size=100,
            multiprocess=60,
            use_cache=True,
            max_tokens=10000,
        )  # type: ignore

        responses = LLMCompletions.get_llm_completions(args, prompts)

        filtered = []
        for commit, response in zip(commits, responses):
            response = response[0]
            reasoning = response.split("[/REASON]")[0].split("[REASON]")[1].strip()
            answer = response.split("[/ANSWER]")[0].split("[ANSWER]")[1].strip()

            if answer.lower() == "yes":
                commit.add_llm_reason(reasoning)
                filtered.append(commit)

            if verbose:
                print(f"Commit Hash: {commit.commit_hash}")
                print(f"Commit Message: {commit.message}")
                print(f"Reasoning: {reasoning}")
                print(f"Answer: {answer}")
                print("\n")

        # run retrieval to get affected files
        retriever = PerfCommitAnalyzer.retrieve_affected_files(filtered, repo_path)

        return filtered, retriever

    @staticmethod
    def retrieve_affected_files(commits: list[PerformanceCommit], repo_path: Path):
        retriever = Retriever(repo_path)
        llm_args = LLMArgs(
            model_name="o3-mini",
            cache_batch_size=100,
            multiprocess=60,
            use_cache=False,
            max_tokens=24000,
        )  # type: ignore
        retriever.retrieve_affected_files(commits, llm_args)
        return retriever

    ######################### LLM-based API Identification #########################

    @staticmethod
    def identify_api_prompt(commit: PerformanceCommit, retriever: Retriever):
        prompt = PERF_IDENTIFY_API_TASK.format(
            diff_text=commit.diff_text, message=commit.message
        )

        if count_tokens(prompt) > MAX_COMMIT_TOKENS:
            diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
            prompt = PERF_IDENTIFY_API_TASK.format(
                diff_text=diff_text, message=commit.message
            )

        tokens_so_far = count_tokens(prompt)

        file_content_prompt = "Some repo files:\n\n"
        for file_name in commit.affected_paths:
            content = retriever.file_content_map[file_name]
            new_content = f"File: {file_name}\n\n```{file_name.split('.')[-1]}\n{content}\n```\n\n"
            tokens_so_far += count_tokens(new_content)
            if tokens_so_far > MAX_OAI_TOKENS + THRESHOLD:
                new_content = (
                    new_content[: MAX_OAI_TOKENS - tokens_so_far - THRESHOLD]
                    + "...(truncated)...\n\n```"
                )
                file_content_prompt += new_content
                break
            file_content_prompt += new_content

        return [
            {
                "role": "system",
                "content": PERF_IDENTIFY_API_SYSTEM + "\n\n" + PERF_IDENTIFY_API_DOCS,
            },
            {"role": "user", "content": file_content_prompt},
            {
                "role": "user",
                "content": prompt,
            },
        ]

    @staticmethod
    def llm_get_apis(commits: list[PerformanceCommit], retriever: Retriever):

        if SKIP_API_ANALYSIS:
            for commit in commits:
                commit.add_apis(["SkippedAPIAnalysis"])
                commit.add_llm_api_reason(
                    "Skipped API analysis likely due to non python repo"
                )
            return

        prompts = [
            PerfCommitAnalyzer.identify_api_prompt(commit, retriever)
            for commit in commits
        ]

        args = LLMArgs(
            model_name="o3-mini",
            cache_batch_size=100,
            multiprocess=60,
            use_cache=False,
            max_tokens=24000,
        )  # type: ignore

        responses = LLMCompletions.get_llm_completions(args, prompts)

        for commit, response in zip(commits, responses):
            response = response[0]
            try:
                reasoning = response.split("[/REASON]")[0].split("[REASON]")[1].strip()
                apis = response.split("[/APIS]")[0].split("[APIS]")[1].strip()
                apis = [api.strip() for api in apis.split(",")]
            except:
                apis = []
                reasoning = "No APIs found"
            commit.add_apis(apis)
            commit.add_llm_api_reason(reasoning)

    ######################### Main Analysis #########################

    @staticmethod
    def get_performance_commits(
        repo_path: Path, no_grep: bool, max_year: int
    ) -> list[PerformanceCommit]:

        base_cmd = ["git", "log", "--pretty=format:%H", "-i"]
        grep_filters = [
            "--grep=perf",
            "--grep=performance",
            "--grep=optimize",
            "--grep=speed up",
            "--grep=speedup",
            "--grep=is slow",
            "--grep=faster",
            "--grep=overhead",
            "--grep=latency",
        ]

        # use grep to cut down commits to process
        if not no_grep:
            print("Using grep to filter commits")
            base_cmd = base_cmd[:3] + grep_filters + ["-i"]

        # get commit hashes
        commit_hashes = run_git_command(base_cmd, cwd=repo_path).splitlines()

        # Parse and process commits
        commits = []
        with Pool() as pool:
            commits = list(
                tqdm(
                    pool.starmap(
                        PerfCommitAnalyzer.process_commit,
                        [
                            (commit_hash, repo_path, max_year)
                            for commit_hash in commit_hashes
                        ],
                    ),
                    total=len(commit_hashes),
                )
            )

        commits = [commit for commit in commits if commit is not None]
        print("# Candidate Commits:", len(commits))

        # ask user if they want to proceed with LLM analysis on XX commits
        if not prompt_yes_no("Proceed with LLM analysis on these commits?"):
            return []

        # LLM Analysis
        filtered, retriever = PerfCommitAnalyzer.llm_analysis(commits, repo_path)
        PerfCommitAnalyzer.llm_get_apis(filtered, retriever)
        print("# LLM Filtered Performance Commits:", len(filtered))

        # get diff stats for each performance commit
        for commit in tqdm(filtered, "Adding stats"):
            commit.add_stats(PerfCommitAnalyzer.parse_diff_for_stats(commit, repo_path))

        return filtered

    @staticmethod
    def analyze_repository(args) -> PerfAnalysis:
        repo_url = args.repo_url
        repo_owner, repo_name = repo_url.split("/")[-2:]
        repo_path = ANALYSIS_REPOS_DIR / repo_name

        # Clone the repository if not alread in ANALYSIS_DIR / "repos"
        if not os.path.exists(repo_path):
            subprocess.run(["git", "clone", repo_url, repo_path])

        performance_commits = PerfCommitAnalyzer.get_performance_commits(
            repo_path, args.no_grep, args.max_year
        )

        return PerfAnalysis(
            repo_url=repo_url,
            repo_owner=repo_owner,
            repo_name=repo_name,
            performance_commits=performance_commits,
        )

    @staticmethod
    def save_analysis(analysis: PerfAnalysis, output_file: Path):
        with open(output_file, "w") as f:
            f.write(analysis.model_dump_json(indent=2))

    @staticmethod
    def load_analysis(input_file: Path) -> PerfAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return PerfAnalysis(**data)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch commits from a repository URL.")
    parser.add_argument("yaml_path", type=str, help="Path to the experiment YAML file.")
    parser.add_argument(
        "--max_year",
        type=int,
        required=False,
        default=None,
        help="Maximum year for commits",
    )
    parser.add_argument(
        "--no-grep",
        action="store_true",
        help="Use grep to filter commits",
    )
    args = parser.parse_args()
    configs = load_exp_config(args.yaml_path)

    args.repo_url = configs["repo_url"]
    if "api_docs" in configs:
        PERF_IDENTIFY_API_DOCS = configs["api_docs"]

    analysis = PerfCommitAnalyzer.analyze_repository(args)

    output_file = ANALYSIS_COMMITS_DIR / f"{analysis.repo_name}_commits.json"
    ANALYSIS_APIS_DIR.mkdir(parents=True, exist_ok=True)
    PerfCommitAnalyzer.save_analysis(analysis, output_file)
