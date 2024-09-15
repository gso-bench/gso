from r2e.paths import EXTRACTED_DATA_DIR
from pyperf.paths import TESTGEN_DIR


class PerfTestGenerator:

    @staticmethod
    def generat(args):
        """Generate performance tests for functions"""
        functions = load_functions(EXTRACTED_DATA_DIR / args.in_file)

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
        tasks = []
        for func in functions:
            task = TestGenTask(func)
            task.prepare()
            tasks.append(task)
        return tasks