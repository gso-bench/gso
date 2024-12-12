# pyperf
Generating performance tests for python repositories


## Installation
1. Install `uv` if you don't have it. This is our recommended way to install the package dependencies.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    <details>
    <summary>Additional Tips</summary>
    Add the source of uv to PATH, `~/.zshrc` for zsh or `~/.bashrc` for bash.

    ```bash
    # Example for zsh
    echo 'source /path/to/uv' >> ~/.zshrc
    # Example for bash
    echo 'source /path/to/uv' >> ~/.bashrc
    ```
    </details>

2. Create a virtual environment, clone the repo, and install it.
    ```bash
    git clone --recursive https://github.com/r2e-project/pyperf.git
    cd pyperf && uv venv && source .venv/bin/activate
    uv sync
    ```

3. [Github API token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) and [OpenAI key](https://platform.openai.com/api-keys) setup.
    ```
    export GHAPI_TOKEN="github_token"
    export OPENAI_API_KEY="openai_key"
    ```

## Usage

### 1. Configure experiments

First pick an experiment ID, usually the repository name (say `repo`) -- you will use this ID to refer to the experiment in following steps. Experiments can be configured using a simple YAML file with the following structure:

```yaml
exp_id: "repo"
repo_url: "https://github.com/username/repo"
py_version: 3.9
target_commit: "main"
install_commands:
    - "uv venv --python 3.9"
    - "source .venv/bin/activate"
    - "uv pip install -e ."
```

You can add the repository URL and custom python version & installation commands. You can also specify `api_docs` and `repo_instr` (free form strings) to specify APIs to focus on during analysis and custom performance test generation tips. If `install_commands` is not provided, a
[default set](https://github.com/r2e-project/pyperf/blob/7b65c8fd7d41ae4d46e889d912e4fbc931871f39/src/pyperf/data/problem.py#L5-L6) is used.



### 2. Commit Analysis Pipeline

The [analysis/](/src/pyperf/analysis/) directory contains the performance commit analysis pipeline. It identifies and analyzes performance-related commits in Python repositories and then maps them to high-level APIs that are affected by the changes. More details in the [readme](/src/pyperf/analysis/README.md).

Run the pipeline on any repository using the `commits.py` script:
```bash
python src/pyperf/analysis/commits.py /path/to/experiment.yaml
python src/pyperf/analysis/apis.py repo
```

The commit analysis results are saved as a JSON file in `ANALYSIS_DIR/commits/repo_commits.json`. Then, the API analysis results are saved in `ANALYSIS_DIR/apis/repo_ac_map.json`. You can use the `--no-grep` flag to disable the grep-based filtering and the `--max_year` flag to filter commits by year.


### 3. Generate performance tests

Run the following to generate performance tests for the configured experiment:
```bash
python src/pyperf/generate/generate.py /path/to/experiment.yaml
```

Remember to set [`GHAPI_TOKEN`](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) env var. Creates an experiment workspace in `EXPERIMENTS_DIR/{exp_id}` and moves your configuration file there. It then generates performance tests for the configured experiment and saves it in the workspace as `{exp_id}_problems.json`.

### 4. Execute performance tests

*Prerequisite*: Cloud credentials set up for `skypilot` to spin up machines.
Run `sky check` and follow the instructions it provides to set up credentials.
Then, run the following to execute the generated performance tests:
```bash
python src/pyperf/execute/execute.py --exp_id repo --machines K
```

This runs performance tests for the configured experiment on `K` machines and saves results in the workspace in `{exp_id}_results.json`. Optionally use `--api` to run tests for a single API. Use `--interactive` to run tests in interactive mode (for debugging).

View the stats of the results using:
```bash
python src/pyperf/execute/evaluate.py --exp_id repo
```


<details>
    <summary>Some helpful `skypilot` commands for test runs:</summary>
    
    # view machines running
    sky status
    
    # stream logs or just the status of what's running in a machine
    sky logs machine_name
    sky logs --status machine_name
    
    # ssh into a machine
    ssh machine_name
    
    # shutdown machines
    sky down machine_name
    sky down --all
</details>
