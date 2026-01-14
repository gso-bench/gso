<!-- <p align="center">
  <a href="https://huggingface.co/datasets/gso-bench/gso">
    <img src="https://raw.githubusercontent.com/gso-bench/gso/main/app/static/gso_logo.svg" style="height: 10em" alt="GSO Benchmark" />
  </a>
</p> -->

<!-- <p align="center"><strong>[&nbsp;<a href="https://github.com/gso-bench/gso">Read the Docs</a>&nbsp;]</strong></p> -->

<!-- <p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.12+-1f425f.svg?color=purple">
    </a>
    <a href="LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
    <a href="https://pypi.org/project/gso/">
        <img src="https://img.shields.io/badge/pypi-v0.1.0-blue">
    </a>
</p> -->

---

<h1 align="center">GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents</h1>

GSO (Global Software Optimization) is a benchmark for evaluating language models' capabilities in developing high-performance software. We present 100+ challenging optimization tasks across 10 codebases spanning diverse domains and programming languages. Each task provides a codebase and performance test as a precise specification, with agents required to optmize the codebase and measured against expert developer commits.

## 📰 News
* **[Dec 23, 2025]**: Released evaluation logs and transcripts w/ [Docent](https://transluce.org/docent) support: [gso-bench/gso-experiments](https://github.com/gso-bench/gso-experiments).
* **[Nov 3, 2025]**: Released GSO's HackDetector that catches models reward hacking: [GSO Blog](https://gso-blog.notion.site/gso-hackdetector).
* **[May 30, 2025]**: 🤗 GSO dataset is now available on HuggingFace! Access it at [gso-bench/gso](https://huggingface.co/datasets/gso-bench/gso).
* **[May 30, 2025]**: Prebuilt docker images for GSO tasks are now available on [Docker Hub](https://hub.docker.com/repository/docker/slimshetty/gso/general).
* **[May 30, 2025]**: Initial release of the GSO benchmark: [gso-bench.github.io](https://gso-bench.github.io/)

## 👋 Overview
GSO evaluates language models on software performance optimization. Each task provides:
- A *codebase* with a specific performance bottleneck
- A *performance test* as a precise specification
- An agent must generate a *patch* that improves runtime efficiency
- Success is measured against expert developer optimizations

To access GSO, copy and run the following code:
```python
from datasets import load_dataset
gso = load_dataset('gso-bench/gso', split='test')
```

## 🚀 Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone --recursive https://github.com/gso-bench/gso.git
cd gso && uv venv && source .venv/bin/activate
uv sync
```

(Additional) Setup [HuggingFace](https://huggingface.co/docs/hub/en/security-tokens) token: 
```
export HF_TOKEN="huggingface_token"
```


## 💽 Usage

### Evaluation Harness

1. **Building Dockers for GSO tasks**:
```bash
docker login

uv run src/gso/harness/prepare_images.py \
    --push_to_registry True \
    --dockerhub_username <dockerhub_username> \
    --dockerhub_repo <dockerhub_repo>
```

2. **Running Evaluations**:
```bash
uv run src/gso/harness/opt_at_k.py \
    --prediction_paths <prediction_path> \
    --timeout 3600 \
    --run_id <run_id> \
    --k 1 \
    --model <modelname>
```

For detailed instructions and options, see the [Harness documentation](src/gso/harness/README.md).

### GSO Collection Framework

The collection framework enables you to create your own GSO tasks through a four-step pipeline:

1. **[Commit Extraction & Filtering](src/gso/collect/README.md#overview)**: Extract performance-related commits using LLMs
2. **[API Identification](src/gso/collect/README.md#2-commit-analysis-pipeline)**: Identify affected high-level APIs for each commit
3. **[Performance Test Generation](src/gso/collect/README.md#3-generate-performance-tests)**: Generate tests for API-Commit pairs
4. **[Test Execution](src/gso/collect/README.md#4-execute-performance-tests)**: Execute tests to identify performance improvements

<!-- Required tokens:
```bash
export GHAPI_TOKEN="github_token"
export OPENAI_API_KEY="openai_key"
export HF_TOKEN="huggingface_token"
``` -->

For detailed instructions and usage, see the [Collection Framework documentation](src/gso/collect/README.md).


## ⬇️ Artifacts
| Datasets | Tools | Dockers |
| - | - | - |
| [💿 GSO](https://huggingface.co/datasets/gso-bench/gso) | [🔧 Evaluation Harness](src/gso/harness/) | [🐳 Docker Hub](https://hub.docker.com/repository/docker/slimshetty/gso/general) |
| | [🔧 Collection Framework](src/gso/collect/README.md) | |

## 💫 Contributions
We welcome contributions from the broader NLP, Machine Learning, and Software Engineering research communities! Please file a new pull request or issue and fill in the corresponding templates accordingly.

## ✍️ Citation & license
MIT license. Check `LICENSE` file.

<!-- If you find our work helpful, please use the following citation: -->
<!-- 
```bibtex
@inproceedings{
    gso2025,
    title={GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents},
    author={GSO Team},
    booktitle={Conference Name},
    year={2025},
    url={https://github.com/gso-bench/gso}
}
``` -->
