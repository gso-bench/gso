import os
import re
import json
import subprocess

from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
from multiprocessing import Pool


from ghapi.core import GhApi
from pyperf.constants import ANALYSIS_DIR
from pyperf.analysis.models import PerformanceCommit, RepositoryAnalysis

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")


class RepoPerformanceAnalyzer:
    @staticmethod
    def run_git_command(cmd: List[str], cwd: str = None) -> str:
        return subprocess.check_output(cmd, cwd=cwd, universal_newlines=True).strip()

    @staticmethod
    def process_commit(commit_hash: str, repo_path: str) -> PerformanceCommit:
        # commit subject
        subject = RepoPerformanceAnalyzer.run_git_command(
            ["git", "show", "--no-patch", "--format=%s", commit_hash], cwd=repo_path
        )

        # commit message
        message = RepoPerformanceAnalyzer.run_git_command(
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
        date_str = RepoPerformanceAnalyzer.run_git_command(
            ["git", "show", "-s", "--format=%cd", commit_hash], cwd=repo_path
        )
        date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")

        # changed files
        files_changed = RepoPerformanceAnalyzer.run_git_command(
            ["git", "show", "--name-only", "--format=", commit_hash], cwd=repo_path
        ).split("\n")

        return PerformanceCommit(
            commit_hash=commit_hash,
            subject=subject,
            message=message,
            date=date,
            files_changed=files_changed,
        )

    @staticmethod
    def get_performance_commits(repo_path: str) -> List[PerformanceCommit]:
        # use grep to cut down commits to process
        commit_hashes = RepoPerformanceAnalyzer.run_git_command(
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
                        RepoPerformanceAnalyzer.process_commit,
                        [(commit_hash, repo_path) for commit_hash in commit_hashes],
                    ),
                    total=len(commit_hashes),
                )
            )

        performance_commits = [commit for commit in performance_commits if commit]
        print("# Performance Commits:", len(performance_commits))

        return performance_commits

    @staticmethod
    def analyze_repository(repo_name: str, repo_path: str) -> RepositoryAnalysis:
        performance_commits = RepoPerformanceAnalyzer.get_performance_commits(repo_path)
        return RepositoryAnalysis(
            repo_name=repo_name, performance_commits=performance_commits
        )

    @staticmethod
    def save_analysis(analysis: RepositoryAnalysis, output_file: str):
        with open(output_file, "w") as f:
            f.write(analysis.model_dump_json(indent=2))

    @staticmethod
    def load_analysis(input_file: str) -> RepositoryAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return RepositoryAnalysis(**data)


# Example usage
if __name__ == "__main__":
    repo_name = "numpy"
    repo_path = ANALYSIS_DIR / "repos" / repo_name
    output_file = ANALYSIS_DIR / f"{repo_name}_commits.json"

    analysis = RepoPerformanceAnalyzer.analyze_repository(repo_name, repo_path)
    RepoPerformanceAnalyzer.save_analysis(analysis, output_file)

    # To load the analysis later
    # loaded_analysis = RepoPerformanceAnalyzer.load_analysis(output_file)
