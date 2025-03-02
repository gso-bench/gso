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
from pyperf.generate.args import PerfExpGenArgs


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, args):
        self.config = load_exp_config(args.yaml_path)
        self.exp_id = self.config["exp_id"]
        self.repo = Repo.from_url(self.config["repo_url"])

        # set repo-specific instructions
        if "repo_instr" in self.config:
            self.repo.repo_instr = self.config["repo_instr"]

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
            num_workers=16,
            progress_bar_desc="Preparing tests",
        )

        problems, payloads = [], []
        for output in outputs:
            if output.is_success():
                prob = output.result
                problems.append(prob)
                for test in prob.tests:
                    if args.model_name in ["o1-mini", "o3-mini", "o1-preview"]:
                        prompt = "\n\n".join(
                            msg["content"] for msg in test.chat_messages
                        )
                        payloads.append([{"role": "user", "content": prompt}])
                    else:
                        payloads.append(test.chat_messages)
            else:
                logger.error(f"Failed to prepare: {output.exception_tb}")

        if args.model_name in ["o1-mini", "o3-mini", "o1-preview"]:
            payloads = [p for p in payloads for _ in range(args.n)]
            outputs = LLMCompletions.get_llm_completions(args, payloads)
            outputs = [item for sublist in outputs for item in sublist]
            grouped_outputs = []
            for i in range(0, len(outputs), args.n):
                grouped_outputs.append(outputs[i : i + args.n])

            results = get_generated_tests(grouped_outputs)
        else:
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


if __name__ == "__main__":
    args = fire.Fire(PerfExpGenArgs.parse)
    generator = PerfExpGenerator(args)
    generator.gen(args)
