import fire
from collections import defaultdict

from r2e.llms.completions import LLMCompletions
from r2e.multiprocess import run_tasks_in_parallel
from gso.data import Repo, Problem
from gso.logger import logger
from gso.constants import EXPS_DIR

from gso.utils.io import *
from gso.collect.generate.prompt import *
from gso.collect.generate.helpers import *
from gso.collect.generate.context import prepare_oversample
from gso.collect.generate.args import OversampleArgs
from gso.collect.pids import TEST_PROBLEMS


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, args):
        self.exp_dir = EXPS_DIR / "oversample"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        exp_ids = TEST_PROBLEMS.keys() if args.exp_id is None else [args.exp_id]
        all_problems = [
            problem
            for eid in exp_ids
            for problem in load_problems((EXPS_DIR / f"{eid}" / f"{eid}_results.json"))
        ]
        selected_api_commit_pairs = set(
            [item for sublist in TEST_PROBLEMS.values() for item in sublist]
        )
        selected_pids = set(pid for pid, _ in selected_api_commit_pairs)
        problems = [p for p in all_problems if p.pid in selected_pids]

        if args.api:
            problems = [p for p in problems if p.api == args.api]
            if not problems:
                raise ValueError(f"No problem found for API: {args.api}")

        self.problems = [p for p in problems if p.is_valid()]

        self.api_commit_map = defaultdict(list)
        for pid, commit_hash in selected_api_commit_pairs:
            self.api_commit_map[pid].append(commit_hash)

        # remove commits from prob if not selected
        for prob in self.problems:
            prob.filter_commit_hashes(self.api_commit_map[prob.pid])

        print("Loaded problems: ", sum([len(v) for v in self.api_commit_map.values()]))
        print("Loaded unique APIs: ", len(self.api_commit_map))

    def oversample(self, args) -> list[Problem]:
        logger.debug(f"Oversampling perftests for {len(self.problems)} problems")

        # prepare for test generation
        outputs = run_tasks_in_parallel(
            prepare_oversample,
            [
                (prob.repo, prob, self.api_commit_map[prob.pid])
                for prob in self.problems
            ],
            use_progress_bar=True,
            num_workers=16,
            progress_bar_desc="Preparing context",
        )

        problems, payloads = [], []
        for output in outputs:
            if output.is_success():
                prob: Problem = output.result
                problems.append(prob)
                for test in prob.tests:
                    if args.model_name in ["o1-mini", "o3-mini", "o4-mini"]:
                        prompt = "\n\n".join(
                            msg["content"] for msg in test.chat_messages
                        )
                        payloads.append([{"role": "user", "content": prompt}])
                    else:
                        payloads.append(test.chat_messages)
            else:
                logger.error(f"Failed to prepare: {output.exception_tb}")

        print(f"Generating {len(problems)} problems with {len(payloads)} payloads")

        if args.model_name in ["o1-mini", "o3-mini", "o4-mini"]:
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
            for test in prob.tests:
                if idx < len(results):
                    test.add_samples(results[idx])
                    idx += 1

            prob.clear_results()  # clear all previous results if any

        save_problems(self.exp_dir / f"oversample_problems.json", problems)


if __name__ == "__main__":
    args = fire.Fire(OversampleArgs.parse)
    generator = PerfExpGenerator(args)
    generator.oversample(args)
