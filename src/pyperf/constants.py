import os
from pathlib import Path
from pathlib import Path
import yaml

current_dir = Path(__file__).parent

# --------- Path Constants ---------

HOME_DIR = Path(os.path.expanduser("~"))
PYPERF_BUCKET_DIR = HOME_DIR / "buckets" / "pyperf_bucket"

ANALYSIS_DIR = PYPERF_BUCKET_DIR / "analysis"
ANALYSIS_REPOS_DIR = ANALYSIS_DIR / "repos"
ANALYSIS_COMMITS_DIR = ANALYSIS_DIR / "commits"
ANALYSIS_APIS_DIR = ANALYSIS_DIR / "apis"

EXPS_DIR = PYPERF_BUCKET_DIR / "experiments"
SKYGEN_TEMPLATE = current_dir / "execute" / "template.yaml"
PHASE1_TEMPLATE = current_dir / "execute" / "phase1.txt"
PHASE2_TEMPLATE = current_dir / "execute" / "phase2.txt"

# --------- Harness/Evals Constants ---------

SUBMISSIONS_DIR = PYPERF_BUCKET_DIR / "submissions"
DATASET_DIR = PYPERF_BUCKET_DIR / "datasets"
INSTANCE_IMAGE_BUILD_DIR = Path("logs/build_images/instances")
RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")
EVALUATION_REPORTS_DIR = Path("reports")

# --------- Build Constants ---------

MIN_PROB_SPEEDUP = 1.2  # min speedup to consider a problem as a benchmark instance
MAX_TEST_COUNT = 15  # max number of tests to run per problem
LONG_RUNNING_MAX_TEST_COUNT = 5  # max number of tests to run per long runtime problem

# --------- Grading Constants ---------

# min % speedup to consider a patch as optimized
OPTIM_THRESH_PERC = 15

# min factor to consider a patch as optimized
OPTIM_THRESH_FACTOR = 1 / (1 - (OPTIM_THRESH_PERC / 100))
