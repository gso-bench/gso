# Collection Framework 


## Overview

1. **Commit Extraction & Filtering**: Extracts potential performance-related commits from a given repository using LLMs.
2. **API Identification**: Use RAG w/ LLM pipeline to identify affected high-level APIs for each performance commit.
3. **Performance Test Generation**: Generates performance tests for the identified API-Commit pairs using LLMs.
4. **Test Execution**: Execute performane tests and identify problems (API-Commit pairs) that show performance improvements.

**Prerequisite**: 

Install [r2e](https://github.com/r2e-project/r2e) for llm helpers and parallel sampling:
```
uv pip install git+https://github.com/r2e-project/r2e
```

Setup [Github](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens), [OpenAI](https://platform.openai.com/api-keys), [HuggingFace](https://huggingface.co/docs/hub/en/security-tokens) tokens
```
export GHAPI_TOKEN="github_token"
export OPENAI_API_KEY="openai_key"
export HF_TOKEN="huggingface_token"
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
[default set](https://github.com/r2e-project/gso/blob/7b65c8fd7d41ae4d46e889d912e4fbc931871f39/src/gso/data/problem.py#L5-L6) is used. See examples in the [experiments/](/exps/) directory.



### 2. Commit Analysis Pipeline

The [analysis/](/src/gso/collect/analysis/) directory contains the performance commit analysis pipeline. It identifies and analyzes performance-related commits in Python repositories and then maps them to high-level APIs that are affected by the changes. More details in the [readme](/src/gso/collect/analysis/README.md).

Run the pipeline on any repository using the `commits.py` script:
```bash
python src/gso/collect/analysis/commits.py /path/to/experiment.yaml
python src/gso/collect/analysis/apis.py repo
```

The commit analysis results are saved as a JSON file in `ANALYSIS_DIR/commits/repo_commits.json`. Then, the API analysis results are saved in `ANALYSIS_DIR/apis/repo_ac_map.json`. You can use the `--no-grep` flag to disable the grep-based filtering and the `--max_year` flag to filter commits by year.


### 3. Generate performance tests

Run the following to generate performance tests for the configured experiment:
```bash
python src/gso/collect/generate/generate.py /path/to/experiment.yaml
```

Remember to set [`GHAPI_TOKEN`](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) env var. Creates an experiment workspace in `EXPERIMENTS_DIR/{exp_id}` and moves your configuration file there. It then generates performance tests for the configured experiment and saves it in the workspace as `{exp_id}_problems.json`.

### 4. Execute performance tests

*Prerequisite*: Cloud credentials set up for `skypilot` to spin up machines.
Run `sky check` and follow the instructions it provides to set up credentials.
Then, run the following to execute the generated performance tests:
```bash
python src/gso/collect/execute/execute.py --exp_id repo --machines K
```

This runs performance tests for the configured experiment on `K` machines and saves results in the workspace in `{exp_id}_results.json`. Optionally use `--api` to run tests for a single API. Use `--interactive` to run tests in interactive mode (for debugging).

View the stats of the results using:
```bash
python src/gso/collect/execute/evaluate.py --exp_id repo
```