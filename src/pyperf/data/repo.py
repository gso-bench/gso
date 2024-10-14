from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyperf.data import PerformanceCommit


class RepositoryAnalysis(BaseModel):
    repo_url: str
    repo_owner: str
    repo_name: str
    performance_commits: list["PerformanceCommit"] = Field(default_factory=list)
