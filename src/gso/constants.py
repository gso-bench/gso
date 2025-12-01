import os
from pathlib import Path
from pathlib import Path
import yaml

current_dir = Path(__file__).parent

# --------- Path Constants ---------

HOME_DIR = Path(os.path.expanduser("~"))
GSO_BUCKET_DIR = HOME_DIR / "buckets" / "gso_bucket"

ANALYSIS_DIR = GSO_BUCKET_DIR / "analysis"
ANALYSIS_REPOS_DIR = ANALYSIS_DIR / "repos"
ANALYSIS_COMMITS_DIR = ANALYSIS_DIR / "commits"
ANALYSIS_APIS_DIR = ANALYSIS_DIR / "apis"

EXPS_DIR = GSO_BUCKET_DIR / "experiments"
SKYGEN_TEMPLATE = current_dir / "collect" / "execute" / "template.yaml"
PHASE1_TEMPLATE = current_dir / "collect" / "execute" / "phase1.txt"
PHASE2_TEMPLATE = current_dir / "collect" / "execute" / "phase2.txt"

# --------- Harness/Evals Constants ---------

SUBMISSIONS_DIR = GSO_BUCKET_DIR / "submissions"
DATASET_DIR = GSO_BUCKET_DIR / "datasets"
INSTANCE_IMAGE_BUILD_DIR = Path("logs/build_images/instances")
RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")
EVALUATION_REPORTS_DIR = Path("reports")
PLOTS_DIR = Path("plots")

# --------- Build Constants ---------
MIN_PROB_SPEEDUP = 1.2  # min speedup to consider a problem as a benchmark instance
MAX_TEST_COUNT = 20  # max number of tests to run per problem
LOW_TEST_IDEAL_TEST_COUNT = 5  # target test count with low test count
LOW_TEST_FALLBACK_SPEEDUP = 1.1  # min speedup for problems with low test count

# --------- Grading Constants ---------
OPT_THRESH = 0.95  # min speedup to consider as `matching or exceeding` commit perf
HIGH_RESOURCE_REPOS = ["abetlen/llama-cpp-python", "huggingface/tokenizers"]
