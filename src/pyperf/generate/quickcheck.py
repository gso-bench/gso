"""Quickcheck module for running performance tests in a temporary directory w/ a local python environment

NOTE: this can be used to run unsafe code, so be careful with the input!!!!
"""

from pathlib import Path
import subprocess
import shutil
import os
import sys
from typing import Optional, List, Tuple
import tempfile

from pyperf.constants import HOME_DIR
from pyperf.data import Problem


def _run_command(
    cmd: str, cwd: Path, venv_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """Run a shell command in the given directory with optional venv activation"""
    try:
        env = os.environ.copy()
        if venv_path:
            venv_bin = venv_path / "bin"
            env["PATH"] = f"{str(venv_bin)}{os.pathsep}{env['PATH']}"
            env["VIRTUAL_ENV"] = str(venv_path)
            env.pop("PYTHONHOME", None)

        result = subprocess.run(
            cmd.split(), cwd=cwd, env=env, capture_output=True, text=True
        )

        if result.returncode != 0:
            print(result.stderr)
            return False, result.stderr

        if result.stdout:
            print(result.stdout)

        return True, result.stdout

    except Exception as e:
        print(f"Error running '{cmd}': {str(e)}")
        return False, f"Error running '{cmd}': {str(e)}"


def quickcheck(prob: Problem) -> tuple[bool, tuple[str, str]]:
    print(f"=================== QUICKCHECK: {prob.pid} ===================")
    tmp_dir = HOME_DIR / "quickcheck_tmp"
    tmp_dir.mkdir(exist_ok=True)
    repo_dir = tmp_dir / prob.repo.repo_name
    venv_dir = tmp_dir / ".venv"

    print("Temp directory:", tmp_dir)

    # move test to a file in the tmp directory
    test_file = tmp_dir / "test.py"
    with open(test_file, "w") as f:
        f.write(prob.test)

    print("Test:\n")
    print(prob.test)

    # Clone repository if it doesn't exist
    if not repo_dir.exists():
        success, output = _run_command(
            f"git clone {prob.repo.repo_url} {prob.repo.repo_name}", tmp_dir
        )
        if not success:
            return False, output

    # Run each install command from prob.install_commands
    for cmd in prob.install_commands:
        print("Running:", cmd)
        if "uv venv" in cmd:
            success, output = _run_command(cmd, tmp_dir)

        elif "source" in cmd:
            continue

        else:
            success, output = _run_command(cmd, repo_dir, venv_dir)

        if not success:
            return False, output

    # run the test
    success, output = _run_command(f"python test.py results_a.txt", tmp_dir, venv_dir)
    shutil.rmtree(tmp_dir)

    if not success:
        return False, output
    else:
        return True, None

    print(f"==================== END ===================")
