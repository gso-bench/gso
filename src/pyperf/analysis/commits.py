import os
import re
import json
import argparse
from pathlib import Path

from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from ghapi.core import GhApi

from r2e.llms.llm_args import LLMArgs
from r2e.llms.completions import LLMCompletions

from pyperf.constants import ANALYSIS_DIR
from pyperf.data.models import PerformanceCommit, RepositoryAnalysis
from pyperf.analysis.parser import DiffParser
from pyperf.analysis.retriever import Retriever
from pyperf.analysis.prompt import *
from pyperf.analysis.utils import *

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")
MAX_COMMIT_TOKENS = 20000
MAX_OAI_TOKENS = 90000
THRESHOLD = 200


class PerfCommitAnalyzer:
    @staticmethod
    def parse_diff_for_stats(commit: PerformanceCommit) -> dict[str, int]:
        parser = DiffParser()
        diff = parser.parse_diff(
            commit.old_commit_hash,
            commit.commit_hash,
            commit.diff_text,
            commit.message,
            commit.date,
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
            "is_bugfix": diff.is_bugfix,
            "is_feature": diff.is_feature,
            "is_refactor": diff.is_refactor,
            "commit_year": diff.commit_date.year,
        }

        return stats

    @staticmethod
    def process_commit(commit_hash: str, repo_path: Path) -> PerformanceCommit | None:
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

        # changed files
        files_changed = run_git_command(
            ["git", "show", "--name-only", "--format=", commit_hash], cwd=repo_path
        ).split("\n")

        # commit diff
        old_commit_hash = f"{commit_hash}^"
        diff_text = run_git_command(
            ["git", "diff", "-p", old_commit_hash, commit_hash], cwd=repo_path
        )

        return PerformanceCommit(
            commit_hash=commit_hash,
            subject=subject,
            message=message,
            date=date,
            files_changed=files_changed,
            diff_text=diff_text,
        )

    ######################### LLM-based Commit Filtering #########################

    @staticmethod
    def analysis_prompt(commit: PerformanceCommit) -> str:
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
            model_name="gpt-4o-mini",
            cache_batch_size=100,
            multiprocess=30,
            use_cache=True,
        )

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
            model_name="gpt-4o-mini",
            cache_batch_size=100,
            multiprocess=30,
            use_cache=True,
        )
        retriever.retrieve_affected_files(commits, llm_args)
        return retriever

    ######################### LLM-based API Identification #########################

    @staticmethod
    def identify_api_prompt(commit: PerformanceCommit, retriever: Retriever) -> str:
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
                "content": PERF_IDENTIFY_API_SYSTEM,
            },
            {"role": "user", "content": file_content_prompt},
            {
                "role": "user",
                "content": prompt,
            },
        ]

    @staticmethod
    def llm_get_apis(commits: list[PerformanceCommit], retriever: Retriever):
        prompts = [
            PerfCommitAnalyzer.identify_api_prompt(commit, retriever)
            for commit in commits
        ]

        args = LLMArgs(
            model_name="gpt-4o-mini",
            cache_batch_size=100,
            multiprocess=30,
            use_cache=True,
        )

        responses = LLMCompletions.get_llm_completions(args, prompts)

        for commit, response in zip(commits, responses):
            response = response[0]
            apis = response.split("[/APIS]")[0].split("[APIS]")[1].strip()
            apis = [api.strip() for api in apis.split(",")]
            commit.add_apis(apis)

    ######################### Main Analysis #########################

    @staticmethod
    def get_performance_commits(repo_path: Path) -> list[PerformanceCommit]:
        # use grep to cut down commits to process
        commit_hashes = run_git_command(
            [
                "git",
                "log",
                "--pretty=format:%H",
                "--grep=perf",
                "--grep=performance",
                "--grep=optimize",
                "--grep=speed up",
                "--grep=speedup",
                "--grep=is slow",
                "-i",
            ],
            cwd=repo_path,
        ).splitlines()

        # Parse and process commits
        commits = []
        with Pool() as pool:
            commits = list(
                tqdm(
                    pool.starmap(
                        PerfCommitAnalyzer.process_commit,
                        [(commit_hash, repo_path) for commit_hash in commit_hashes],
                    ),
                    total=len(commit_hashes),
                )
            )

        print("# Candidate Commits:", len(commits))

        # LLM Analysis
        filtered, retriever = PerfCommitAnalyzer.llm_analysis(commits, repo_path)
        PerfCommitAnalyzer.llm_get_apis(filtered, retriever)
        print("# LLM Filtered Performance Commits:", len(filtered))

        # get diff stats for each performance commit
        for commit in filtered:
            commit.add_stats(PerfCommitAnalyzer.parse_diff_for_stats(commit))

        return filtered

    @staticmethod
    def analyze_repository(args) -> RepositoryAnalysis:
        repo_url = args.repo_url
        repo_owner, repo_name = repo_url.split("/")[-2:]
        repo_path = ANALYSIS_DIR / "repos" / repo_name

        # Clone the repository if not alread in ANALYSIS_DIR / "repos"
        if not os.path.exists(repo_path):
            subprocess.run(["git", "clone", repo_url, repo_path])

        performance_commits = PerfCommitAnalyzer.get_performance_commits(repo_path)

        return RepositoryAnalysis(
            repo_url=repo_url,
            repo_owner=repo_owner,
            repo_name=repo_name,
            performance_commits=performance_commits,
        )

    @staticmethod
    def save_analysis(analysis: RepositoryAnalysis, output_file: Path):
        with open(output_file, "w") as f:
            f.write(analysis.model_dump_json(indent=2))

    @staticmethod
    def load_analysis(input_file: Path) -> RepositoryAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return RepositoryAnalysis(**data)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch commits from a repository URL.")
    parser.add_argument("repo_url", type=str, help="The URL of the repository")
    args = parser.parse_args()

    analysis = PerfCommitAnalyzer.analyze_repository(args)

    output_file = ANALYSIS_DIR / "commits" / f"{analysis.repo_name}_commits.json"
    PerfCommitAnalyzer.save_analysis(analysis, output_file)

    # To load the analysis later
    # loaded_analysis = PerfCommitAnalyzer.load_analysis(output_file)
