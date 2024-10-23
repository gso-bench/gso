# pyperf
Generating performance tests for python repositories


## Installation
1. Install `uv` if you don't have it. This is our recommended way to install the package dependencies.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment, clone the repo, and install it.
```bash
git clone https://github.com/r2e-project/pyperf.git
cd pyperf && uv venv && source .venv/bin/activate
uv sync
```

## Usage

### 1. Commit Analysis Pipeline

The [analysis/](/src/pyperf/analysis/) directory contains the performance commit analysis pipeline. It identifies and analyzes performance-related commits in Python repositories and then maps them to high-level APIs that are affected by the changes. More details in the [readme](/src/pyperf/analysis/README.md).

Run the pipeline on any repository using the `commits.py` script:
```bash
python src/pyperf/analysis/commits.py https://github.com/username/repo.git
```

The analysis results are saved as a JSON file in the `ANALYSIS_DIR/commits/` directory with the format `<repo>_commits.json`.


### 2. Configure experiments

<!-- TODO: make this a different command from generate -->
First pick an experiment ID, usually the repository name (say `repo`). Then run the following command that will generate an experiment configuration file:

```bash
python src/pyperf/generate/generate.py --exp_id repo
```

The outout is saved in `EXPERIMENTS_DIR/{exp_id}/{exp_id}.yaml`:
```yaml
repo_url: ""
candidates:
    - api: ""
      base_commit: ""
```

You can now manually add the repository URL and the candidate API(s) that you want to generate and execute performance tests for. You can also add multiple candidate APIs.


### 3. Generate performance tests

Run the following to generate performance tests for the configured experiment:
```bash
python src/pyperf/generate/generate.py --exp_id repo
```

The generated performance tests are saved in `EXPERIMENTS_DIR/{exp_id}/{exp_id}_problems.json` directory.


### 4. Execute performance tests

Run the following to execute the generated performance tests:
```bash
python src/pyperf/execute/execute.py --exp_id repo --machines K
```

This will run performance tests for the configured experiment on `K` machines and save results in `EXPERIMENTS_DIR/{exp_id}/{exp_id}_results.json`. Optionally use `-api` to run tests for a specific API.