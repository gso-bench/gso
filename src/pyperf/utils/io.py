import json
import yaml
from pathlib import Path
from pyperf.data import Problem
from pyperf.constants import EXPS_DIR


def load_json(file_path: str | Path) -> dict | list:
    """Load a JSON file from disk."""
    with open(file_path, "r") as f:
        return json.load(f)


def load_problems(file_path) -> list[Problem]:
    problems_data = load_json(file_path)
    problems = [Problem(**problem) for problem in problems_data]
    return problems


def save_problems(file_path, problems: list[Problem]):
    problems_data = [problem.dict() for problem in problems]

    with open(file_path, "w") as f:
        json.dump(problems_data, f, indent=4)


def load_experiment(exp_id="temp", api=None) -> dict:
    """Load an experiment from disk."""
    exp_dir = EXPS_DIR / exp_id
    exp_path = exp_dir / f"{exp_id}.yaml"

    EXPS_DIR.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        print(f"Experiment dir for {exp_id} does not exists. Creating one.")
        exp_dir.mkdir()

    if not exp_path.exists():
        print(f"Experiment file {exp_id} does not exist. Creating one.")
        exp_path.touch()

        # add default experiment data
        _default_exp_str = (
            'repo_url: ""\n' "candidates:\n" '  - api: ""\n' '    base_commit: ""\n'
        )
        with open(exp_path, "w") as f:
            f.write(_default_exp_str)

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
