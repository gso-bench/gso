import os
import re
import json
import subprocess
import argparse
from pathlib import Path

from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
from multiprocessing import Pool


from ghapi.core import GhApi
from pyperf.constants import ANALYSIS_DIR

from pyperf.analysis.data.models import PerformanceCommit, RepositoryAnalysis
from pyperf.analysis.parser import DiffParser

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")


class PerfCommitAnalyzer:
    @staticmethod
    def run_git_command(cmd: List[str], cwd: Path | None = None) -> str:
        return subprocess.check_output(
            cmd, cwd=cwd, universal_newlines=True, errors="replace"
        ).strip()

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
        subject = PerfCommitAnalyzer.run_git_command(
            ["git", "show", "--no-patch", "--format=%s", commit_hash], cwd=repo_path
        )

        # commit message
        message = PerfCommitAnalyzer.run_git_command(
            ["git", "show", "--no-patch", "--format=%B", commit_hash], cwd=repo_path
        )

        # filter out irrelevant commits if keyword not in message
        perf_keywords = ["perf", "performance", "optimize", "speed up", "speedup"]
        if not any(
            re.search(r"\b" + keyword + r"\b", message, re.IGNORECASE)
            for keyword in perf_keywords
        ):
            return None

        # commit date
        date_str = PerfCommitAnalyzer.run_git_command(
            ["git", "show", "-s", "--format=%cd", commit_hash], cwd=repo_path
        )
        date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")

        # changed files
        files_changed = PerfCommitAnalyzer.run_git_command(
            ["git", "show", "--name-only", "--format=", commit_hash], cwd=repo_path
        ).split("\n")

        # commit diff
        old_commit_hash = f"{commit_hash}^"
        diff_text = PerfCommitAnalyzer.run_git_command(
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

    @staticmethod
    def get_performance_commits(repo_path: Path) -> List[PerformanceCommit]:
        # use grep to cut down commits to process
        commit_hashes = PerfCommitAnalyzer.run_git_command(
            [
                "git",
                "log",
                "--pretty=format:%H",
                "--grep=perf",
                "--grep=performance",
                "--grep=optimize",
                "--grep=speed up",
                "--grep=speedup",
                "-i",
            ],
            cwd=repo_path,
        ).splitlines()

        print("# Candidates:", len(commit_hashes))

        performance_commits = []
        with Pool() as pool:
            performance_commits = list(
                tqdm(
                    pool.starmap(
                        PerfCommitAnalyzer.process_commit,
                        [(commit_hash, repo_path) for commit_hash in commit_hashes],
                    ),
                    total=len(commit_hashes),
                )
            )

        performance_commits = [commit for commit in performance_commits if commit]
        print("# Performance Commits:", len(performance_commits))

        # get diff stats for each performance commit
        for commit in performance_commits:
            commit.add_stats(PerfCommitAnalyzer.parse_diff_for_stats(commit))

        return performance_commits

    @staticmethod
    def analyze_repository(
        repo_url: str, repo_owner: str, repo_name: str, repo_path: Path
    ) -> RepositoryAnalysis:
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

    repo_url = args.repo_url
    repo_owner, repo_name = repo_url.split("/")[-2:]
    repo_path = ANALYSIS_DIR / "repos" / repo_name
    output_file = ANALYSIS_DIR / "commits" / f"{repo_name}_commits.json"

    # Clone the repository if not alread in ANALYSIS_DIR / "repos"
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo_url, repo_path])

    analysis = PerfCommitAnalyzer.analyze_repository(
        repo_url, repo_owner, repo_name, repo_path
    )
    PerfCommitAnalyzer.save_analysis(analysis, output_file)

    # To load the analysis later
    # loaded_analysis = PerfCommitAnalyzer.load_analysis(output_file)
