from collections import defaultdict
from pathlib import Path


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
