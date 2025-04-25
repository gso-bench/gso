from collections import defaultdict
from pathlib import Path
import os


def zip_results(results_dir: Path):
    """Zip the results directory by mapping corresponding files together"""

    file_groups = defaultdict(dict)

    for filename in results_dir.iterdir():
        if filename.suffix == ".txt" or filename.suffix == ".json":
            parts = filename.stem.split("_", 2)
            if len(parts) >= 3:
                file_type, identifier = parts[0], parts[1] + "_" + parts[2]
                file_groups[identifier][file_type] = results_dir / filename

    return file_groups


def add_tokens_to_installs(install_commands):
    """Add tokens to a problem's install commands if set in the environment"""
    install_commands.append(
        f"export HF_TOKEN={os.getenv('HF_TOKEN')}" if os.getenv("HF_TOKEN") else ""
    )
    return install_commands
