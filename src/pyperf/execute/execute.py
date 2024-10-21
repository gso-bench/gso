import shutil
import argparse

from pyperf.execute.skymgr import SkyManager
from pyperf.utils.io import load_problems
from pyperf.constants import SKYGEN_TEMPLATE, TESTGEN_DIR
from pyperf.data import Problem
from pyperf.execute.harness import TEST_HARNESS

problems = [
    {
        "pid": "networkx-timing-test",
        "cloud": "gcp",
        "region": "us-central1",
        "instance_type": "n2-standard-16",
        "repo_url": "https://github.com/huggingface/datasets",
        "repo_name": "datasets",
        "before_commit": "d9a08aafc21ad0b8d70e6d149353e380ed01ca12^1",
        "after_commit": "main",
        "setup_commands": [
            "sudo apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev",
            "sudo apt-get install -y libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk",
            "sudo apt-get install -y libharfbuzz-dev libfribidi-dev libxcb1-dev libx11-dev",
        ],
        "install_commands": [
            "uv venv --python 3.9",
            "source .venv/bin/activate",
            "which python",
            "python --version",
            "uv pip install -e .",
            "uv pip install requests",
            "uv pip show datasets",
        ],
        "test": TEST_HARNESS,
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute tasks with SkyManager")
    parser.add_argument(
        "-m", "--machines", type=int, default=1, help="Number of machines to use"
    )
    args = parser.parse_args()

    # TODO: load problems from a file
    # problems = load_problems(TESTGEN_DIR / "test.json")
    problem = Problem(**problems[0])

    yaml_template = SkyManager.load_template(SKYGEN_TEMPLATE)
    wspace = SkyManager.create_workspace(problem, yaml_template)

    for i in range(args.machines):
        SkyManager.launch_task(
            f"{problem.pid}_task.yaml", wspace, cluster=f"sky-pyperf-{i}"
        )
        results = SkyManager.get_results(wspace, cluster=f"sky-pyperf-{i}")
        print(results)

    SkyManager.cleanup_workspace(wspace)
    SkyManager.cleanup_all_clusters()
