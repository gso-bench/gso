import shutil

from pyperf.execute.skymgr import SkyManager
from pyperf.utils.io import load_problems
from pyperf.constants import SKYGEN_TEMPLATE, TESTGEN_DIR

if __name__ == "__main__":
    data = load_problems(TESTGEN_DIR / "test.json")
    problem = data[0]

    yaml_template = SkyManager.load_template(SKYGEN_TEMPLATE)
    wspace = SkyManager.create_workspace(problem, yaml_template)
    SkyManager.launch_task(f"{problem.pid}_task.yaml", wspace)
    SkyManager.cleanup_workspace(wspace)
