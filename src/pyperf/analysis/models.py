from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class PerformanceCommit(BaseModel):
    commit_hash: str
    subject: str
    message: str
    date: datetime
    files_changed: List[str] = Field(default_factory=list)
    functions_changed: List[str] = Field(default_factory=list)


class RepositoryAnalysis(BaseModel):
    repo_url: str
    repo_owner: str
    repo_name: str
    performance_commits: List[PerformanceCommit] = Field(default_factory=list)
