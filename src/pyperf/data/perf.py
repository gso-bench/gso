import re
from datetime import datetime
from pydantic import BaseModel, Field


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

    @property
    def old_commit_hash(self) -> str:
        return f"{self.commit_hash}^"

    @property
    def linked_pr(self) -> str:
        # TODO(@manish): expand this to other patterns if necessary
        pr_pattern = r"\(#(\d+)\)"
        match = re.search(pr_pattern, self.subject)
        return match.group(1) if match else None

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

    def add_affected_paths(self, paths: list[str]):
        self.affected_paths.extend(paths)


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
