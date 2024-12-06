import fire

from r2e.llms.completions import LLMCompletions
from r2e.multiprocess import run_tasks_in_parallel
from pyperf.data import Repo, Problem
from pyperf.logger import logger
from pyperf.constants import EXPS_DIR, ANALYSIS_APIS_DIR

from pyperf.utils.io import *
from pyperf.generate.prompt import *
from pyperf.generate.helpers import *
from pyperf.generate.context import prepare
from pyperf.generate.quickcheck import quickcheck
from pyperf.generate.args import PerfExpGenArgs


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, args):
        self.config = load_exp_config(args.yaml_path)
        self.exp_id = self.config["exp_id"]
        self.repo = Repo.from_url(self.config["repo_url"])
        self.candidates = self.get_commit_map(self.repo)
        self.exp_dir = EXPS_DIR / self.exp_id

    def get_commit_map(self, repo: Repo):
        """Get the api-commit map for the repository"""
        ac_map = load_map(ANALYSIS_APIS_DIR / f"{repo.repo_name}_ac_map.json")
        return ac_map.api_to_commits

    def gen(self, args) -> list[Problem]:
        logger.debug(f"Generating perftests: {self.repo}")

        # create new problems
        problems = [
            Problem.create_prob(self.repo, api, commits, self.config)
            for api, commits in self.candidates.items()
        ]

        # filter their commits to a maximum year and remove empty problems
        for prob in problems:
            prob.filter_commits(args.max_year)
        problems = [prob for prob in problems if prob.num_commits() > 0]

        # prepare each problem for test generation
        outputs = run_tasks_in_parallel(
            prepare,
            [(self.repo, prob) for prob in problems],
            use_progress_bar=True,
            progress_bar_desc="Preparing tests",
        )

        problems, payloads = [], []
        for output in outputs:
            if output.is_success():
                prob = output.result
                problems.append(prob)
                for test in prob.tests:
                    payloads.append(test.chat_messages)
            else:
                logger.error(f"Failed to prepare: {output.exception_tb}")

        outputs = LLMCompletions.get_llm_completions(args, payloads)
        results = get_generated_tests(outputs)
        idx = 0
        for prob in problems:
            existing_tests = prob.tests
            for test in prob.tests:
                if idx < len(results):
                    test.add_samples(results[idx])
                    idx += 1

        save_problems(self.exp_dir / f"{self.exp_id}_problems.json", problems)
        return problems

    def genquickcheck(self, args, max_rounds: int = 1) -> list[Problem]:
        """Debug tests with iterative improvement based on quickcheck feedback"""
        logger.debug(f"Generating perftests with quickcheck feedback: {self.repo}")

        problems = [
            Problem.create_prob(self.repo, api, commits, self.config)
            for api, commits in self.candidates.items()
        ]

        outputs = run_tasks_in_parallel(
            prepare,
            [(self.repo, prob) for prob in problems],
            use_progress_bar=True,
            progress_bar_desc="Preparing tests",
        )

        problems = []
        for output in outputs:
            if output.is_success():
                problem = output.result
                problems.append(problem)
            else:
                logger.error(f"Failed to prepare: {output.exception_tb}")
                continue

        if args.n > 1:
            raise ValueError("Quickcheck only supports generating 1 test per problem")

        for problem in problems:
            current_round = 0
            while current_round < max_rounds:
                prob_test = problem.tests[0]
                payloads = [prob_test.chat_messages]  # only use first test
                outputs = LLMCompletions.get_llm_completions(args, payloads)
                test = get_generated_tests(outputs)[0][0]  # get first generated test
                prob_test.add_sample(test)
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


if __name__ == "__main__":
    args = fire.Fire(PerfExpGenArgs.parse)
    generator = PerfExpGenerator(args)
    if args.quickcheck:
        generator.genquickcheck(args)
    else:
        generator.gen(args)
