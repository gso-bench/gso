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
    py_version: str = Field(default="3.9", description="Python version")

    # commit info
    base_commit: str = Field(..., description="Commit hash for before test")
    target_commit: str = Field(default="main", description="Commit hash for after test")

    # commands
    setup_commands: list[str] = Field(init=False, default=[])
    install_commands: list[str] = Field(init=False, default=[])

    chat_messages: list[dict[str, str]] = Field(default=[], description="Chat messages")
    test: str = Field(default="if __name__ == '__main__': pass", description="Test")
    results: dict[int, dict[str, str]] = Field(default={}, description="Exec Results")

    def model_post_init(self, __context) -> None:
        if self.setup_commands == []:
            self.setup_commands = [
                "sudo apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev",
                "sudo apt-get install -y libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk",
                "sudo apt-get install -y libharfbuzz-dev libfribidi-dev libxcb1-dev libx11-dev",
            ]

        if self.install_commands == []:
            self.install_commands = [
                f"uv venv --python {self.py_version}",
                "source .venv/bin/activate",
                "which python",
                "python --version",
                "uv pip install -e .",
                "uv pip install requests",
                f"uv pip show {self.repo.repo_name}",
            ]

    # helpers to generate test

    def init_chat(self, sys_msg: str, context_msg: str, task_msg: str):
        self.chat_messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": context_msg},
            {"role": "user", "content": task_msg},
        ]

    def add_test(self, test: str):
        self.test = test

    def add_result(self, key: int, result: dict[str, str]):
        self.results[key] = result

    # helper to create a problem from a dict

    @classmethod
    def create_prob(cls, repo: Repo, cand: dict, config: dict):
        api = cand["api"]
        base_commit = cand["base_commit"]
        pid = repo.repo_name + "-" + api + "-" + base_commit[:7]
        return cls(pid=pid, repo=repo, api=api, **config, base_commit=base_commit)
