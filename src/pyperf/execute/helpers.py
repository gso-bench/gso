import random
import traceback

from pyperf.models import FunctionUnderTest, MethodUnderTest
from pyperf.execute.service import ServiceManager


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
        return self_equiv_futs([fut], conn)
    except Exception as e:
        tb = traceback.format_exc()
        pass
    finally:
        simulator.stop_container()
        conn.close()

    fut.test_history.update_exec_stats({"error": tb})
    print(f"Error@{fut.repo_id}:\n{tb}")

    return False, tb, fut
