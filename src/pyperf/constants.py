import os
from pathlib import Path
from pathlib import Path
import yaml

# Load configuration from config.yaml
current_dir = Path(__file__).parent
config_path = current_dir / "config.yml"
with open(config_path, "r") as file:
    config: dict[str, str] = yaml.safe_load(file)  # type: ignore


HOME_DIR = Path(os.path.expanduser("~"))
PYPERF_BUCKET_DIR = HOME_DIR / config["pyperf_bucket_dir"]

# Note: the repos and extracted_data directories are borrowed from r2e
REPOS_DIR = HOME_DIR / "buckets" / "local_repoeval_bucket" / "repos"

# the following directories are specific to pyperf:
TESTGEN_DIR = PYPERF_BUCKET_DIR / "testgen"
ANALYSIS_DIR = PYPERF_BUCKET_DIR / "analysis"
ANALYSIS_REPOS_DIR = ANALYSIS_DIR / "repos"
