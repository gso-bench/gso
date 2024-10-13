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
cd pyperf
uv venv
source .venv/bin/activate
uv sync
```
