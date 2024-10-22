from r2e.llms.completions import LLMCompletions
from pyperf.analysis.commits import PerfCommitAnalyzer
from pyperf.data import Repo, Problem, PerformanceCommit
from pyperf.utils.io import load_experiment
from pyperf.logger import logger

from pyperf.generate.prompt import *
from pyperf.generate.harness import TEST_HARNESS
from pyperf.generate.helpers import *

from pydantic import BaseModel, Field, HttpUrl


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, config: dict):
        self.repo = Repo.from_url(config["repo_url"])
        self.candidates = config["candidates"]

    def gen(self) -> list[Problem]:
        """Propose performance test experiments for APIs"""
        logger.debug(f"Generating perftest: {self.repo}")
        problems = [self.prepare_prob(cand) for cand in self.candidates]
        return problems

    def quickcheck():
        pass

    def prepare_prob(self, cand) -> Problem:
        prob = Problem.create_prob(self.repo, cand)
        commit = PerfCommitAnalyzer.process_commit(
            prob.base_commit, self.repo.local_repo_path
        )
        context_msg = CONTEXT_MSG.format(
            api=prob.api,
            repo_name=self.repo.repo_name,
            commit_message=strip_empty_lines(commit.message),
            commit_diff=commit.diff_text,
        )
        context_msg += PR_INFO.format(
            pr_messages=get_github_convo(self.repo, commit.linked_pr)
        )

        task_msg = f"Write a test for the {prob.api} API in the {self.repo.repo_name} repository"
        prob.init_chat(SYSTEM_MSG, context_msg, task_msg)

        return prob


if __name__ == "__main__":
    exp_id = "exp"

    # load the yaml config
    exp_config = load_experiment(exp_id)
    generator = PerfExpGenerator(exp_config)
    problems = generator.gen()
