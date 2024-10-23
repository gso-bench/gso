from pydantic import BaseModel, Field, HttpUrl
from pyperf.data.repo import Repo


class Problem(BaseModel):
    pid: str = Field(default="test", description="ID of the problem")
    repo: Repo = Field(..., description="Repository info")
    api: str = Field(..., description="API to test")

    # vm info
    cloud: str = Field(default="gcp", description="Cloud provider")
    region: str = Field(default="us-central1", description="Cloud region")
    instance_type: str = Field(default="n2-standard-16", description="Instance type")

    # commit info
    base_commit: str = Field(..., description="Commit hash for before test")
    target_commit: str = Field(default="main", description="Commit hash for after test")

    # commands
    setup_commands: list[str] = Field(
        default_factory=list, description="Setup commands to run"
    )
    install_commands: list[str] = Field(
        default_factory=list, description="Install commands to run"
    )

    chat_messages: list[dict[str, str]] = Field(default=[], description="Chat messages")
    test: str = Field(
        default="if __name__ == '__main__': pass", description="Test code to run"
    )

    # helpers to generate test

    def init_chat(self, sys_msg: str, context_msg: str, task_msg: str):
        self.chat_messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": context_msg},
            {"role": "user", "content": task_msg},
        ]

    def add_test(self, test: str):
        self.test = test

    # helper to create a problem from a dict

    @classmethod
    def create_prob(cls, repo: Repo, data: dict):
        api = data["api"]
        base_commit = data["base_commit"]
        pid = repo.repo_name + "-" + api + "-" + base_commit[:7]
        return cls(pid=pid, repo=repo, api=api, base_commit=base_commit)
