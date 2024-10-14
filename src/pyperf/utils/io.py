import json
from pathlib import Path
from pyperf.data import Problem


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
