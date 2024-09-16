import fire

from r2e.paths import EXTRACTED_DATA_DIR
from r2e.multiprocess import run_tasks_in_parallel_iter
from r2e.llms.completions import LLMCompletions

from pyperf.constants import TESTGEN_DIR
from pyperf.generate.task import TestGenTask
from pyperf.generate.args import PerfTestGenArgs
from pyperf.generate.context import get_context_wrapper
from pyperf.generate.utils import get_generated_tests, timestamp
from pyperf.models import Tests, create_code_under_test
from pyperf.utils.data import load_functions, write_functions_under_test


class PerfTestGenerator:

    @staticmethod
    def generate(args):
        """Generate performance tests for functions"""
        functions = load_functions(EXTRACTED_DATA_DIR / args.in_file)
        functions = functions[:4]
        tasks = PerfTestGenerator.prepare_tasks(functions)
        payloads = [task.chat_messages for task in tasks]
        outputs = LLMCompletions.get_llm_completions(args, payloads)
        results = get_generated_tests(outputs)
        futs = [create_code_under_test(func) for func in functions]

        for fut, test in zip(futs, results):
            fut.update_history(
                Tests(
                    tests={"test_0": test},
                    gen_model=args.model_name,
                    gen_date=timestamp(),
                )
            )

        TESTGEN_DIR.mkdir(parents=True, exist_ok=True)
        write_functions_under_test(futs, TESTGEN_DIR / f"{args.exp_id}_generate.json")

    @staticmethod
    def prepare_tasks(functions) -> list[TestGenTask]:
        """Prepare tasks for generating tests"""

        # 1: generate context for each function
        context_gen_tasks = [
            (args.context_type, func, args.max_context_size) for func in functions
        ]
        context_iter = run_tasks_in_parallel_iter(
            get_context_wrapper,
            context_gen_tasks,
            num_workers=8,
            use_progress_bar=True,
            progress_bar_desc="Generating contexts",
        )

        # 2: create test generation tasks
        tasks = []
        for func, task_result in zip(functions, context_iter):
            if task_result.is_success():
                context = task_result.result
                func.add_context(context)
                tasks.append(TestGenTask(func_meth=func))
            else:
                print(f"Error generating context:\n{task_result.exception_tb}")

        return tasks


if __name__ == "__main__":
    args = fire.Fire(PerfTestGenArgs)
    PerfTestGenerator.generate(args)
