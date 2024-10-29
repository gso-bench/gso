import time
import shutil
import argparse

from pyperf.execute.skymgr import SkyManager
from pyperf.utils.io import load_problems, save_problems
from pyperf.constants import SKYGEN_TEMPLATE, EXPS_DIR
from pyperf.data import Problem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute tasks with SkyManager")
    parser.add_argument("-e", "--exp_id", type=str, help="Experiment ID", required=True)
    parser.add_argument("-a", "--api", type=str, help="Specific API", required=False)
    parser.add_argument("-m", "--machines", type=int, default=1, help="# machines")
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()

    exp_dir = EXPS_DIR / f"{args.exp_id}"
    problems = load_problems(exp_dir / f"{args.exp_id}_problems.json")

    if args.api:
        prob: Problem = [p for p in problems if p.api == args.api][0]
    else:
        # TODO: add support to run all APIs in the experiment
        prob = problems[0]

    yaml_template = SkyManager.load_template(SKYGEN_TEMPLATE)
    wspace = SkyManager.create_workspace(prob, yaml_template)
    queue = [f"sky-pyperf-{args.exp_id}-{i}" for i in range(args.machines)]

    for c in queue:
        SkyManager.launch_task(
            f"{prob.pid}_task.yaml", wspace, cluster=c, interactive=args.interactive
        )

    # Poll for completion
    while queue:
        for c in queue:
            if SkyManager.is_complete(workspace=wspace, cluster=c):
                queue.remove(c)
        print(f"Waiting for {len(queue)} tasks to complete...")
        time.sleep(10)

    # Get results
    for i in range(args.machines):
        res_str, result = SkyManager.get_results(
            wspace, cluster=f"sky-pyperf-{args.exp_id}-{i}"
        )
        prob.add_result(key=i, result=result)
        print(res_str)

    save_problems(exp_dir / f"{args.exp_id}_results.json", problems)
    SkyManager.cleanup_workspace(wspace)
    SkyManager.cleanup_all_clusters()
