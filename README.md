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

The analysis results are saved as a JSON file in `ANALYSIS_DIR/commits/repo_commits.json`.


### 2. Configure experiments

First pick an experiment ID, usually the repository name (say `repo`) -- you will use this ID to refer to the experiment in following steps. Experiments can be configured using a simple YAML file with the following structure:

```yaml
exp_id: "repo"
repo_url: "https://github.com/username/repo.git"
candidates:
    - api: "abc.XYZ"
      base_commit: "commit_hash"
py_version: 3.9
install_commands:
    - "uv venv --python 3.9"
    - "source .venv/bin/activate"
    - "uv pip install -e ."
```

You can add the repository URL, the candidate repo API(s) to generate tests for, and custom python version & installation commands. If `install_commands` is not provided, a
[default set](https://github.com/r2e-project/pyperf/blob/7b65c8fd7d41ae4d46e889d912e4fbc931871f39/src/pyperf/data/problem.py#L5-L6) is used.


### 3. Generate performance tests

Run the following to generate performance tests for the configured experiment:
```bash
python src/pyperf/generate/generate.py /path/to/experiment.yaml
```

Creates an experiment workspace in `EXPERIMENTS_DIR/{exp_id}` and moves your configuration file there. It then generates performance tests for the configured experiment and saves it in the workspace as `{exp_id}_problems.json`.

### 4. Execute performance tests

Run the following to execute the generated performance tests:
```bash
python src/pyperf/execute/execute.py --exp_id repo --machines K
```

This runs performance tests for the configured experiment on `K` machines and saves results in the workspace in `{exp_id}_results.json`. Optionally use `--api` to run tests for a single API.