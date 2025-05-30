import argparse
import ast
import sys
import os
from collections import Counter

from gso.constants import *
from gso.data import Problem
from gso.utils.io import load_problems

SYSTEM_LIBS = set()
if hasattr(sys, "stdlib_module_names"):
    SYSTEM_LIBS = set(sys.stdlib_module_names)


def extract_imports(problem: Problem, lib: str):
    imports_count = Counter()

    for test in problem.tests:
        for sample in test.samples:
            tree = ast.parse(sample)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if (
                            not alias.name.startswith(lib)
                            and alias.name not in SYSTEM_LIBS
                        ):
                            imp = alias.name.split(".")[0]
                            imports_count[imp] += 1
                elif isinstance(node, ast.ImportFrom):
                    if (
                        not node.module.startswith(lib)
                        and node.module not in SYSTEM_LIBS
                    ):
                        imp = node.module.split(".")[0]
                        imports_count[imp] += 1

    return imports_count


def main(exp_id, lib):
    exp_dir = EXPS_DIR / exp_id
    all_problems = load_problems(exp_dir / f"{exp_id}_problems.json")

    total_imports = Counter()
    for problem in all_problems:
        total_imports.update(extract_imports(problem, lib))

    # Print ranked imports with their frequencies
    print("Dependencies to installed (ranked by freq):")
    for module, count in total_imports.most_common():
        print(f"- {module}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Suggest dependencies to install for a given experiment"
    )
    parser.add_argument("-e", "--exp_id", type=str, help="Experiment ID", required=True)
    parser.add_argument(
        "-l",
        "--lib",
        type=str,
        help="import name for library under test",
        required=True,
    )

    args = parser.parse_args()

    main(args.exp_id, args.lib)
