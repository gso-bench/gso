import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl
from gso.constants import ANALYSIS_REPOS_DIR


class Repo(BaseModel):
    repo_url: str
    repo_owner: str
    repo_name: str
    repo_instr: str | None = Field(default=None)

    @property
    def local_repo_path(self):
        local_repo_path = ANALYSIS_REPOS_DIR / self.repo_name

        if not os.path.exists(local_repo_path):
            subprocess.run(["git", "clone", self.repo_url, local_repo_path])

        return local_repo_path

    @property
    def full_name(self):
        return f"{self.repo_owner}/{self.repo_name}"

    # constructor from url
    @classmethod
    def from_url(cls, url: str):
        repo_owner, repo_name = url.split("/")[-2:]
        return cls(repo_url=url, repo_owner=repo_owner, repo_name=repo_name)
