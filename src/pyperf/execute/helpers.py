import json
import rpyc
import random
import traceback

from pyperf.models import FunctionUnderTest, MethodUnderTest
from pyperf.execute.service import ServiceManager
from pyperf.execute.utils import get_fut_data


def run_fut_with_port_mp(args) -> tuple[bool, str, FunctionUnderTest | MethodUnderTest]:
    fut: FunctionUnderTest | MethodUnderTest = args
    port = random.randint(3000, 10000)
    output = run_fut_with_port(fut, port)
    return output


def run_fut_with_port(
    fut: FunctionUnderTest | MethodUnderTest, port: int
) -> tuple[bool, str, FunctionUnderTest | MethodUnderTest]:
    try:
        simulator, conn = ServiceManager.get_service(fut.repo_id, port, local=True)
    except Exception as e:
        print("Service error@", fut.repo_id, repr(e))
        fut.test_history.update_exec_stats({"error": repr(e)})
        return False, repr(e), fut

    try:
        return exec_perf_futs([fut], conn)
    except Exception as e:
        tb = traceback.format_exc()
        pass
    finally:
        if simulator:
            simulator.stop_container()
        conn.close()

    fut.test_history.update_exec_stats({"error": tb})
    print(f"Error@{fut.repo_id}:\n{tb}")

    return False, tb, fut


def exec_perf_futs(
    futs: list[FunctionUnderTest | MethodUnderTest], conn: rpyc.Connection
) -> tuple[bool, str, FunctionUnderTest | MethodUnderTest]:
    """Executes performance tests for given futs via the service client

    Args:
        futs (list[FunctionUnderTest | MethodUnderTest]): list of functions under test
        conn (rpyc.Connection): connection to the service client

    Returns:
        tuple[bool, str, FunctionUnderTest | MethodUnderTest]: success, error, fut
    """
    service = conn.root
    assert service is not None, "Test service is None"

    repo_data, fut_data, test_data = get_fut_data(futs, local=True)

    ####### Setup the service #######
    service.setup_repo(repo_data)
    service.setup_function(fut_data)
    service.setup_test(test_data)

    init_response = service.init()
    init_output = str(init_response["output"])
    init_error = str(init_response["error"])

    # NOTE hacks to ignore invalid errors:
    # - escapes in docstrings / nameerors due to type_checking=false
    # TODO remove once python versions are set according to repo)
    ignore_patterns = ["SyntaxWarning: invalid escape sequence", "NameError: "]

    if init_error and not any(p in init_error for p in ignore_patterns):
        futs[0].test_history.update_exec_stats(
            {"output": init_output, "error": init_error}
        )
        print(f"Init Error@{futs[0].id}:\n{init_error}\n\n")
        return False, init_error, futs[0]

    ####### Execute the performance test #######

    try:
        submit_response = service.submit()
    except Exception as e:
        futs[0].test_history.update_exec_stats({"error": repr(e)})
        print(f"Submit Error@{futs[0].id}:\n{repr(e)}\n\n")
        return False, repr(e), futs[0]

    submit_error = str(submit_response["error"])

    if "logs" not in submit_response:
        futs[0].test_history.update_exec_stats({"error": submit_error})
        print(f"Submit Error@{futs[0].id}:\n{submit_error}\n\n")
        return False, submit_error, futs[0]

    submit_logs = json.loads(submit_response["logs"])
    submit_logs["output"] = submit_response["output"]
    futs[0].test_history.update_exec_stats(submit_logs)

    valids = [x["valid"] for x in submit_logs["run_tests_logs"].values()]
    return all(valids), submit_error, futs[0]
