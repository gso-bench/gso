from collections import defaultdict

from pydantic import BaseModel, Field, HttpUrl
from pyperf.generate.harness import TEST_HARNESS, TIMEIT_TEMPLATE
from pyperf.data.repo import Repo
from pyperf.data.perf import PerformanceCommit


class Tests(BaseModel):
    commit_hash: str = Field(..., description="Commit hash on which tests are run")
    chat_messages: list[dict[str, str]] = Field(default=[], description="Chat messages")
    samples: list[str] = Field(default=[], description="Sampled tests")

    def init_chat(self, sys_msg: str, context_msg: str, task_msg: str):
        self.chat_messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": context_msg},
            {"role": "user", "content": task_msg},
        ]

    def add_sample(self, test: str):
        self.samples.append(test + TIMEIT_TEMPLATE + TEST_HARNESS)

    def add_samples(self, samples: list[str]):
        self.samples.extend([t + TIMEIT_TEMPLATE + TEST_HARNESS for t in samples])

    @property
    def quick_hash(self) -> str:
        return self.commit_hash[:7]

    # helper to create a test for a commit
    @classmethod
    def from_commit(cls, commit):
        return cls(commit_hash=commit.commit_hash)


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
    base_commit: str = Field(default="", description="Base Commit for problem")
    target_commit: str = Field(default="main", description="Target Commit for problem")

    # commands
    setup_commands: list[str] = Field(init=False, default=[])
    install_commands: list[str] = Field(init=False, default=[])

    commits: list[PerformanceCommit] = []
    tests: list[Tests] = []

    # key: machine_id, value: list of results from multiple runs (run = commit + test pair)
    results: dict[int | str, list[dict]] = defaultdict(list)

    def model_post_init(self, __context) -> None:
        if self.setup_commands == []:
            self.setup_commands = [
                "sudo apt update -y && sudo upt upgrade -y",
                "sudo apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev",
                "sudo apt-get install -y libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk",
                "sudo apt-get install -y libharfbuzz-dev libfribidi-dev libxcb1-dev libx11-dev",
                "sudo apt install -y gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config",
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

    def set_tests(self, tests: list[Tests]):
        """Set test objects for the problem"""
        self.tests = tests

    def add_results(self, key: int, results: list[dict]):
        """Add execution results for the problem"""
        self.results[key] = results

    def set_base_commit(self, commit_hash: str):
        """Set the final base commit for this problem"""
        self.base_commit = commit_hash

    def filter_commits(self, max_year: int):
        """Filter out commits older than max_year"""
        self.commits = [c for c in self.commits if c.date.year >= max_year]

    def get_test(self, commit_hash: str, test_id: str) -> str:
        """Get test object for a commit"""
        for test in self.tests:
            if test.quick_hash == commit_hash:
                return test.samples[test_id]
        return None

    # helper to get properties of the problem
    def num_commits(self) -> int:
        return len(self.commits)

    def num_tests(self) -> int:
        return sum(len(test.samples) for test in self.tests)

    def num_results(self) -> int:
        return sum(len(res) for res in self.results.values())

    def num_runs(self) -> int:
        return len(self.results)

    def mach1_res(self) -> list[dict]:
        if not self.results:
            return []

        first_key = list(self.results.keys())[0]
        return self.results[first_key]

    def is_valid(self) -> bool:
        return self.num_results() > 0

    def num_valid_commits(self) -> list[PerformanceCommit]:
        return len({res["commit"] for res in self.mach1_res()})

    def num_valid_tests(self) -> list[str]:
        return len({res["test_file"] for res in self.mach1_res()})

    # helper to create a problem from a dict

    @classmethod
    def create_prob(cls, repo: Repo, api: str, commits: list, config: dict):
        pid = repo.repo_name.lower() + "-" + api.lower()
        return cls(pid=pid, repo=repo, api=api, commits=commits, **config)
