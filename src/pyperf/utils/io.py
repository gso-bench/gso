import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path

from pyperf.data import Problem, APICommitMap
from pyperf.constants import EXPS_DIR


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: any) -> any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def load_json(file_path: str | Path) -> dict | list:
    """Load a JSON file from disk."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_map(file_path) -> APICommitMap:
    ac_map = load_json(file_path)
    return APICommitMap(**ac_map)


def load_problems(file_path) -> list[Problem]:
    problems_data = load_json(file_path)
    problems = [Problem(**problem) for problem in problems_data]
    return problems


def save_problems(file_path, problems: list[Problem]):
    problems_data = [problem.dict() for problem in problems]

    with open(file_path, "w") as f:
        json.dump(problems_data, f, indent=4, cls=DateTimeEncoder)


# Custom dumper to manage indentation
class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def load_exp_config(yaml_path, api=None) -> dict:
    """Load an experiment from disk."""
    # load the local yaml and get the exp_id
    with open(yaml_path, "r") as f:
        local_file = yaml.safe_load(f)
        exp_id = local_file["exp_id"]

    # create experiments directory and experiment file
    exp_dir = EXPS_DIR / exp_id
    exp_path = exp_dir / f"{exp_id}.yaml"
    EXPS_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(exist_ok=True)

    # copy the local yaml to the experiments directory
    print(f"Copying experiment to {exp_path}")
    shutil.copy(yaml_path, exp_path)

    ##############################

    print(f"Loading experiment from {exp_path}")
    with open(exp_path, "r") as f:
        resp = yaml.safe_load(f)

    # if a single api is requested, remove all other apis from the experiment
    if api:
        print(f"Finding API {api} in the experiment.")
        api_only = [d for d in resp["candidates"] if d["api"] == api]

        if not api_only:
            print(f"API {api} not found in the experiment. Returning all APIs.")
            return resp

        resp["candidates"] = api_only
    return resp
