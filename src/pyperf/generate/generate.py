import fire

from r2e.llms.completions import LLMCompletions
from pyperf.analysis.commits import PerfCommitAnalyzer
from pyperf.data import Repo, Problem, PerformanceCommit
from pyperf.logger import logger
from pyperf.constants import EXPS_DIR

from pyperf.utils.io import *
from pyperf.generate.prompt import *
from pyperf.generate.helpers import *
from pyperf.generate.quickcheck import quickcheck
from pyperf.generate.args import PerfExpGenArgs
from pyperf.generate.harness import TEST_HARNESS


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, args):
        self.config = load_exp_config(args.yaml_path)
        self.exp_id = self.config["exp_id"]
        self.repo = Repo.from_url(self.config["repo_url"])
        self.candidates = self.config["candidates"]
        self.exp_dir = EXPS_DIR / self.exp_id

    def gen(self, args) -> list[Problem]:
        logger.debug(f"Generating perftests: {self.repo}")

        problems = [self.prepare(cand) for cand in self.candidates]
        payloads = [p.chat_messages for p in problems]
        outputs = LLMCompletions.get_llm_completions(args, payloads)
        results = get_generated_tests(outputs)

        for prob, test in zip(problems, results):
            prob.add_test(test + TEST_HARNESS)

        save_problems(self.exp_dir / f"{self.exp_id}_problems.json", problems)
        return problems

    def genquickcheck(self, args, max_rounds: int = 1) -> list[Problem]:
        """Generate tests with iterative improvement based on quickcheck feedback"""
        logger.debug(f"Generating perftests with quickcheck feedback: {self.repo}")

        problems = [self.prepare(cand) for cand in self.candidates]

        for problem in problems:
            current_round = 0
            while current_round < max_rounds:
                payloads = [problem.chat_messages]
                outputs = LLMCompletions.get_llm_completions(args, payloads)
                test = get_generated_tests(outputs)[0]
                problem.add_test(test + TEST_HARNESS)
                success, error_output = quickcheck(problem)

                if success:
                    logger.info(
                        f"Test for {problem.pid} succeeded on round {current_round + 1}"
                    )
                    break

                if current_round == max_rounds - 1:
                    logger.warning(
                        f"Failed to generate working test for {problem.pid} after {max_rounds} rounds"
                    )
                    break

                # Add feedback and continue to next round
                feedback_msg = FEEDBACK_PROMPT.format(error_output=error_output)
                problem.chat_messages.append({"role": "assistant", "content": test})
                problem.chat_messages.append({"role": "user", "content": feedback_msg})
                current_round += 1

        save_problems(self.exp_dir / f"{self.exp_id}_problems.json", problems)
        return problems

    def prepare(self, cand) -> Problem:
        prob = Problem.create_prob(self.repo, cand, self.config)
        commit = PerfCommitAnalyzer.process_commit(
            prob.base_commit, self.repo.local_repo_path
        )
        context_msg = CONTEXT_MSG.format(
            api=prob.api,
            repo_name=self.repo.repo_name,
            commit_message=strip_empty_lines(commit.message),
            commit_diff=commit.diff_text,
        )

        # Added checking if PR message is None
        if commit.linked_pr is not None:
            pr_messages = get_github_convo(self.repo, commit.linked_pr)
            context_msg += PR_INFO.format(pr_messages=pr_messages)
        else:
            context_msg += "No associated pull request for this commit.\n"

        task_msg = (
            f"Write a test for the {prob.api} API in the {self.repo.repo_name} repository. "
            "Remember to NOT time the setup code."
        )
        prob.init_chat(SYSTEM_MSG, context_msg, task_msg)

        return prob


if __name__ == "__main__":
    args = fire.Fire(PerfExpGenArgs.parse)
    # args.quickcheck = False
    generator = PerfExpGenerator(args)
    if args.quickcheck:
        generator.genquickcheck(args)
    generator.gen(args)
