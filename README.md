# pyperf
Generating performance tests for python repositories


## Installation
```bash
git clone https://github.com/r2e-project/pyperf.git
cd pyperf
pip install -e .
```

<!-- ## Jax Setup

### 1. Prepare the scripts and get the PR summary
```bash
cd pyperf/src/pyperf
uv run prepare_pr.py --repo_full_name google/jax --pr_number 22114 --mode diff --function_name <some_function_in_pr>
```

### 2. Run the repo setup script
```bash

# additional setup steps for jax + GPU
uv pip install tensorstore
uv pip uninstall jax jaxlib
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
export JAX_PLATFORM_NAME=gpu

# for any repo, run the generated setup script
uv run sh logs/google___jax_22114/setup_repo.sh
```

### 3. Run performance tests
```bash
uv run runtests.py
``` -->