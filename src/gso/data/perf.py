import re
import subprocess
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path


class PerformanceCommit(BaseModel):
    commit_hash: str
    subject: str
    message: str
    date: datetime
    files_changed: list[str] = Field(default_factory=list)
    functions_changed: list[str] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
    affected_paths: list[str] = Field(default_factory=list)
    apis: list[str] = Field(default_factory=list)
    diff_text: str = ""
    llm_reason: str = ""
    llm_api_reason: str = ""
    repo_path: Path | None = Field(default=None)

    @property
    def old_commit_hash(self) -> str:
        return f"{self.commit_hash}^"

    @property
    def linked_pr(self) -> str:
        # Find linked pr from commit message
        pr_pattern = r"\(#(\d+)\)"
        match = re.search(pr_pattern, self.subject)
        if match:
            return match.group(1)

        # Find merged pr in nearby direct ancestry path
        try:
            cmd = [
                "git",
                "log",
                "--merges",
                "--ancestry-path",
                "--pretty=%H %s",
                "--reverse",
                f"{self.commit_hash}^..HEAD",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=self.repo_path,
            )

            merge_pattern = r"#(\d+)"
            merge_commits = result.stdout.strip().split("\n")[:10]
            for mc in merge_commits:
                if "Merge pull request" in mc:
                    match = re.search(merge_pattern, mc)
                    if match:
                        return match.group(1)
        except Exception as e:
            pass

        return None

    # TODO: add a linked_issue property
    # @property
    # def linked_issue(self) -> str:
    #     pass

    def add_stat(self, key: str, value: int):
        self.stats[key] = value

    def add_stats(self, stats: dict[str, int]):
        self.stats.update(stats)

    def add_llm_reason(self, reason: str):
        self.llm_reason = reason

    def add_apis(self, apis: list[str]):
        self.apis = apis

    def add_llm_api_reason(self, reason: str):
        self.llm_api_reason = reason

    def add_affected_paths(self, paths: list[str]):
        self.affected_paths.extend(paths)

    def quick_hash(self) -> str:
        return self.commit_hash[:7]

    def __str__(self):
        return f"{self.quick_hash()}: {self.subject}"


class PerfAnalysis(BaseModel):
    repo_url: str
    repo_owner: str
    repo_name: str
    performance_commits: list[PerformanceCommit]


class APICommitMap(BaseModel):
    repo_url: str
    repo_owner: str
    repo_name: str
    api_to_commits: dict[str, list[PerformanceCommit]]
