import shutil
import argparse

from pyperf.execute.skymgr import SkyManager
from pyperf.utils.io import load_problems
from pyperf.constants import SKYGEN_TEMPLATE, TESTGEN_DIR
from pyperf.data import Problem
from pyperf.execute.harness import TEST_HARNESS

problems = [
    {
        "pid": "pillow-timing-test",
        "cloud": "gcp",
        "region": "us-central1",
        "instance_type": "n2-standard-16",
        "repo_url": "https://github.com/python-pillow/Pillow",
        "repo_name": "Pillow",
        "before_commit": "4554bcb4e425fac461c39e84919bacd0c5a0dbae^1",
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
            "uv pip show pillow",
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
