This directory contains the performance commit analysis phase of the pyperf pipeline. 
It identifies and analyzes performance-related commits in Python repositories.

#### Pipeline Overview

1. **Commit Extraction & Filtering**: Extracts potential performance-related commits from a given repository. Uses an LLM to filter commits and identify truly performance-related changes.
2. **API Identification**: Uses a RAG w/ LLM pipeline to identify affected high-level APIs for each performance commit.
