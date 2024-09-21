# Script to setup a repository for pyperf deployment
REPO_URL="https://github.com/r2e-project/r2e.git"
LOCAL_REPO_PATH="~/buckets/my_repos/r2e"
EXP_ID="demo"
MULTIPROCESS=8
CACHE_BATCH_SIZE=100

# Parse arguments
while getopts u:p:e:m:c: flag
do
    case "${flag}" in
        u) REPO_URL=${OPTARG};;
        p) LOCAL_REPO_PATH=${OPTARG};;
        e) EXP_ID=${OPTARG};;
        m) MULTIPROCESS=${OPTARG};;
        c) CACHE_BATCH_SIZE=${OPTARG};;
    esac
done

REPO_NAME=$(basename $REPO_URL .git)

# Clone the repository and move it to R2E's workspace
if [ ! -d ~/buckets/local_repoeval_bucket/repos/$REPO_NAME ]; then
    git clone $REPO_URL ~/buckets/local_repoeval_bucket/repos/$REPO_NAME
fi

# Extract functions and methods from the repo
python src/r2e/r2e/repo_builder/extract_func_methods.py --exp_id $EXP_ID --overwrite_extracted --disable_all_filters


# cd to the repo and install it
cd ~/buckets/local_repoeval_bucket/repos/$REPO_NAME


### Climt
# uv pip install libpython
# uv pip install -r requirements_dev.txt
# uv run setup.py build_ext --inplace
# uv run setup.py install
# uv pip install xarray
# uv pip install sympl


### astropy
uv pip install -e .