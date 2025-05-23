#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, List, Dict


def export_tests(input_json: Path, log_dir: Path) -> None:
    """
    Export each sample in each test to individual .py files under log_dir/problem_id/test_idx/
    """
    data = json.loads(input_json.read_text())
    log_dir.mkdir(parents=True, exist_ok=True)

    for prob in data:
        pid = prob.get("pid", "unknown_problem")
        prob_dir = log_dir / pid
        # clear existing
        if prob_dir.exists():
            for child in prob_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
        prob_dir.mkdir(parents=True, exist_ok=True)

        for tests in prob.get("tests", []):
            commit = tests.get("commit_hash", "no_hash")
            test_dir = prob_dir / commit
            test_dir.mkdir(parents=True, exist_ok=True)
            for idx, sample in enumerate(tests.get("samples", [])):
                sample_file = test_dir / f"test_{idx}.py"
                sample_file.write_text(sample)
    print(f"Exported tests to {log_dir}")


def import_tests(input_json: Path, log_dir: Path, output_json: Path) -> None:
    """
    Read edited sample files and replace tests.samples accordingly, then write to output_json.
    """
    data = json.loads(input_json.read_text())

    for prob in data:
        pid = prob.get("pid", "unknown_problem")
        prob_dir = log_dir / pid
        for tests in prob.get("tests", []):
            commit = tests.get("commit_hash", "no_hash")
            test_dir = prob_dir / commit
            if not test_dir.exists():
                continue  # no edits
            new_samples: List[str] = []
            # collect files sorted by index
            files = sorted(test_dir.glob("test_*.py"))
            for file in files:
                new_samples.append(file.read_text())
            tests["samples"] = new_samples

    output_json.write_text(json.dumps(data, indent=2))
    print(f"Rebuilt dataset saved to {output_json}")


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: build_manual.py <export|import> <input.json> <logs_dir> [<output.json>]"
        )
        sys.exit(1)

    mode = sys.argv[1]
    input_json = Path(sys.argv[2])
    log_dir = Path(sys.argv[3])

    if mode == "export":
        export_tests(input_json, log_dir)
    elif mode == "import":
        if len(sys.argv) < 5:
            print("For import mode, provide <output.json> as fourth argument.")
            sys.exit(1)
        output_json = Path(sys.argv[4])
        import_tests(input_json, log_dir, output_json)
    else:
        print(f"Unknown mode '{mode}'. Use 'export' or 'import'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
