import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from argparse import ArgumentTypeError
from typing import cast
from datasets import load_dataset, load_from_disk, Dataset

from gso.data import Problem, APICommitMap
from gso.data.dataset import GSOInstance
from gso.constants import EXPS_DIR, DATASET_DIR, EVALUATION_REPORTS_DIR

############## MAIN DATASET ##############


def load_gso_dataset(
    name="gso-bench/gso", split="test", instance_ids=None
) -> list[GSOInstance]:
    """
    Load GSO dataset from Hugging Face Datasets or local .json/.jsonl file
    """
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)

    # Load from local .json/.jsonl file
    if name.endswith(".json") or name.endswith(".jsonl"):
        with open(DATASET_DIR / Path(name)) as f:
            if name.endswith(".jsonl"):
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)
        dataset_ids = {instance["instance_id"] for instance in dataset}
    else:
        # Load from Hugging Face Datasets
        if (Path(name) / split / "dataset_info.json").exists():
            dataset = cast(Dataset, load_from_disk(Path(name) / split))
        else:
            dataset = cast(Dataset, load_dataset(name, split=split))
        dataset_ids = {instance["instance_id"] for instance in dataset}

    if instance_ids:
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [
            instance for instance in dataset if instance["instance_id"] in instance_ids
        ]

    return [GSOInstance(**instance) for instance in dataset]


def load_gso_predictions(predictions_path: str, dataset_name: str, split: str):
    if predictions_path == "gold":
        raise NotImplementedError("Loading gold predictions not implemented yet")

    if predictions_path.endswith(".json"):
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
            if isinstance(predictions, dict):
                predictions = list(predictions.values())
            if not isinstance(predictions, list):
                raise ValueError(
                    "Predictions must be a list[prediction] or a dictionary[instance_id: prediction]"
                )
    elif predictions_path.endswith(".jsonl"):
        with open(predictions_path, "r") as f:
            predictions = [json.loads(line) for line in f]
    else:
        raise ValueError("Predictions path must be .json or .jsonl")

    # Validate that each prediction has an instance_id
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError(f"Each prediction must be a dictionary, got {type(pred)}")
        if "instance_id" not in pred:
            raise ValueError(f"Each prediction must contain '{'instance_id'}'")

    return predictions


############## MODEL EVALUATION REPORTS ##############


def load_report(report_path: str) -> dict:
    """Load a model evaluation report."""
    full_path = EVALUATION_REPORTS_DIR / report_path
    if not full_path.exists():
        raise FileNotFoundError(f"Report not found: {full_path}")

    return load_json(full_path)


############## INTERMEDIATE DATA ##############


class CustomEncoder(json.JSONEncoder):
    def default(self, obj: any) -> any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
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
    existing_problems = {}
    try:
        with open(file_path, "r") as f:
            existing_data = json.load(f)
            existing_problems = {p["pid"]: p for p in existing_data}
    except (FileNotFoundError, json.JSONDecodeError):
        existing_problems = {}

    # update/add new problems
    for problem in problems:
        existing_problems[problem.pid] = problem.dict()

    problems_data = list(existing_problems.values())

    with open(file_path, "w") as f:
        json.dump(problems_data, f, indent=4, cls=CustomEncoder)


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


def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
