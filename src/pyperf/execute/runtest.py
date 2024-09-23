import fire
from tqdm import tqdm
import traceback

from r2e.multiprocess import run_tasks_in_parallel_iter

from pyperf.constants import TESTGEN_DIR
from pyperf.models import FunctionUnderTest, MethodUnderTest
from pyperf.utils.data import load_functions_under_test, write_functions_under_test
from pyperf.execute.args import PerfTestRunArgs
from pyperf.execute.helpers import run_fut_with_port, run_fut_with_port_mp
from pyperf.execute.service import ServiceManager


class PerfTestRunner:
    @staticmethod
    def run(args):
        """Run performance tests for functions"""
        futs = load_functions_under_test(TESTGEN_DIR / args.in_file)
        print(f"Loaded {len(futs)} functions under test")

        new_futs = []
        if args.multiprocess == 0:
            new_futs = PerfTestRunner._run_futs_sequential(futs, args)
        else:
            new_futs = PerfTestRunner._run_futs_parallel(futs, args)

        ServiceManager.shutdown()
        write_functions_under_test(new_futs, TESTGEN_DIR / f"{args.exp_id}_out.json")

    @staticmethod
    def _run_futs_sequential(futs, args):
        new_futs = []
        for i, fut in tqdm(enumerate(futs), desc="Running tests", total=len(futs)):
            port = args.port
            try:
                output = run_fut_with_port(fut, port)
            except Exception as e:
                print(f"Error@{fut.repo_id}:\n{repr(e)}")
                tb = traceback.format_exc()
                print(tb)
                continue
            new_futs.append(output[2])

            if (i + 1) % 20 == 0:
                write_functions_under_test(
                    new_futs, TESTGEN_DIR / f"{args.exp_id}_out.json"
                )

        return new_futs

    @staticmethod
    def _run_futs_parallel(futs, args):
        new_futs = []
        outputs = run_tasks_in_parallel_iter(
            run_fut_with_port_mp,
            futs,
            num_workers=args.multiprocess,
            timeout_per_task=args.timeout_per_task,
            use_progress_bar=True,
        )

        for o in outputs:
            if o.is_success():
                new_futs.append(o.result[2])  # type: ignore
            else:
                print(f"Error: {o.exception_tb}")
        return new_futs


if __name__ == "__main__":
    args = fire.Fire(PerfTestRunArgs)
    PerfTestRunner.run(args)
