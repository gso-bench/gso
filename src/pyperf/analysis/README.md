This directory contains the performance commit analysis phase of the pyperf pipeline. 
It identifies and analyzes performance-related commits in Python repositories.

#### Pipeline Overview

1. **Commit Extraction & Filtering **: Extracts potential performance-related commits from a given repository. Uses an LLM to filter commits and identify truly performance-related changes.
2. **API Identification**: Uses a RAG w/ LLM pipeline to identify affected high-level APIs for each performance commit.

#### Usage

##### 1. Commit Extraction & Filtering

The entry point for this phase is `commits.py`. To analyze a repository:

```bash
python commits.py <repository_url>
```

Example:
```bash
python commits.py https://github.com/huggingface/datasets
```

The analysis results are saved as a JSON file in the `ANALYSIS_DIR/commits/` directory with the format `<repo_name>_commits.json`.

##### 2. API Identification

The entry point for this phase is `api.py`. To identify affected APIs:

```bash
python api.py <repo_name>
```

Example:
```bash
python api.py datasets
```

The analysis results are saved as a JSON file in the `ANALYSIS_DIR/api/` directory with the format `<repo_name>_ac_map.json`.