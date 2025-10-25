"""
Utility functions for qualitative analysis scripts.
"""

import re
import ast
from pathlib import Path
from gso.constants import RUN_EVALUATION_LOG_DIR


def get_run_dir(report_file_path: str) -> str | None:
    """Extract the run directory from a report file path."""
    if not report_file_path:
        return None

    path_parts = Path(report_file_path).parts
    if len(path_parts) >= 5 and path_parts[-3] == "pass":
        return path_parts[-2]
    return None


def load_model_patch(instance_id: str, run_dir: str) -> str | None:
    """Load patch for instance from specified run directory."""
    patch_path = RUN_EVALUATION_LOG_DIR / "pass" / run_dir / instance_id / "patch.diff"
    if patch_path.exists():
        try:
            with open(patch_path, "r") as f:
                return f.read()
        except Exception:
            return None
    return None


def simplify_patch(patch_text: str) -> str:
    """Filter patch to only include source code files."""
    # Non-source file extensions to ignore
    ignore_extensions = {
        "json",
        "txt",
        "md",
        "rst",
        "csv",
        "yaml",
        "yml",
        "toml",
        "ini",
        "cfg",
        "lock",
        "pkl",
        "npy",
        "html",
        "css",
        "js",
        "ts",
        "jsx",
        "tsx",  # web files
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "bmp",
        "ico",
        "svg",
        "pdf",  # media
        "zip",
        "tar",
        "gz",
        "bz2",
        "xz",
        "patch",
        "diff",  # archives
        "gitignore",
        "dockerfile",
        "makefile",
        "cmake",
        "m4",  # build files
        "ipynb",
        "fli",
        "tmpl",
        "build",
        "venv",
        "pyc",
        "pyo",  # misc
    }

    # Directories to ignore
    ignore_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "venv",
        "env",
        ".env",
        "build",
        "dist",
        "target",
        "bin",
        "obj",
    }

    lines = patch_text.split("\n")
    filtered_lines = []
    current_file = None
    skip_current_file = False

    for line in lines:
        # Check if this is a file diff header
        if line.startswith("diff --git"):
            # Extract file path from diff header
            match = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            if match:
                file_path = match.group(1)
                current_file = file_path

                # Check if we should skip this file
                skip_current_file = False

                # Check file extension
                if "." in file_path:
                    ext = file_path.split(".")[-1].lower()
                    if ext in ignore_extensions:
                        skip_current_file = True

                # Check if file is in ignored directory
                if not skip_current_file:
                    for ignore_dir in ignore_dirs:
                        if (
                            f"/{ignore_dir}/" in f"/{file_path}/"
                            or file_path.startswith(f"{ignore_dir}/")
                        ):
                            skip_current_file = True
                            break

                # Always include the diff header
                filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        elif line.startswith("+++") or line.startswith("---"):
            # File info lines - include if not skipping
            if not skip_current_file:
                filtered_lines.append(line)
        elif skip_current_file:
            # Skip all content for ignored files
            continue
        else:
            # Include all other lines for source files
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def simplify_test(test_code: str) -> str:
    """Simplify test code display for reward hack detection."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        # If AST parsing fails, fall back to simple truncation
        return test_code[:3000] + "\n# ... (truncated for brevity)"

    # Functions that represent the three key concepts for reward hack detection
    workload_functions = {"setup"}
    experiment_functions = {"experiment"}
    equivalence_functions = {"store_result", "load_result", "check_equivalence"}

    lines = test_code.split("\n")
    simplified_lines = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name

            if func_name in workload_functions:
                simplified_lines.append(f"\n# WORKLOAD SETUP")
                start_line = node.lineno - 1
                end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else node.lineno
                )
                simplified_lines.extend(lines[start_line:end_line])

            elif func_name in experiment_functions:
                simplified_lines.append(f"\n# PERFORMANCE EXPERIMENT")
                start_line = node.lineno - 1
                end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else node.lineno
                )
                simplified_lines.extend(lines[start_line:end_line])

            elif func_name in equivalence_functions:
                if func_name == "check_equivalence":
                    simplified_lines.append(f"\n# EQUIVALENCE CHECKING")
                start_line = node.lineno - 1
                end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else node.lineno
                )
                simplified_lines.extend(lines[start_line:end_line])

    result = "\n".join(simplified_lines)
    if len(result) > 3000:
        result = result[:3000] + "\n# ... (truncated for brevity)"

    return result
