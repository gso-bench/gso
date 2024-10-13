This directory contains the performance commit analysis phase of the pyperf pipeline. 
It identifies and analyzes performance-related commits in Python repositories.

#### Pipeline Overview

1. **Commit Extraction**: Extracts potential performance-related commits from a given repository.
2. **LLM-based Filtering**: Uses an LLM to filter commits and identify truly performance-related changes.
3. **API Identification**: Uses a RAG w/ LLM pipeline to identify affected high-level APIs for each performance commit.

#### Usage

The entry point for this phase is `commits.py`. To analyze a repository:

```bash
python commits.py <repository_url>
```

Example:
```bash
python commits.py https://github.com/username/repo.git
```

The analysis results are saved as a JSON file in the `ANALYSIS_DIR/commits/` directory with the format `<repo_name>_commits.json`.