import shutil

from pyperf.execute.skygen import SkyGen
from pyperf.utils.io import load_problems
from pyperf.constants import SKYGEN_TEMPLATE

if __name__ == "__main__":
    data = load_problems("./execute/input.json")
    problem = data[0]

    yaml_template = SkyGen.load_template(SKYGEN_TEMPLATE)
    wspace = SkyGen.create_workspace(problem, yaml_template)
    SkyGen.cleanup_workspace(wspace)
