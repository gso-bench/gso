<details>
<summary>Prerequisite: A gso dataset that must contain the following fields per task:</summary>

```python
{
    "instance_id": "str",               # gso task instance ID
    "repo": "str",                      # repository name
    "base_commit": "str",               # base commit hash
    "prob_script": "str",               # problem script for model
    "tests": "List[str]",               # test scripts for eval
    "api": "str",                       # API to optimize (optional)
    "hints_text": "str",                # NL desc. of task (optional)
    "setup_commands": "List[str]",      # setup commands for VMs
    "install_commands": "List[str]",    # install commands for repo
    "created_at": "str",                # gt commit timestamp
    "arch": "str",                      # architecture
    "instance_image_tag": "str",        # docker image tag for task
}
```

</details>
</br>


# 1. Building GSO Docker images

## 1.1 Building from scratch
To push the dockers to dockerhub, you need to login first. Then run the following to build the images and push to dockerhub:
```bash
docker login

uv run src/gso/harness/prepare_images.py --dataset_name <dataset_name> --push_to_registry True --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```

- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--max_workers` can be used to scale up parallel builds
- `--force_rebuild` can be used to force rebuild the images even if they already exist on dockerhub/locally.

## 1.2 Pulling prebuilt images from dockerhub
You can also simply pull the prebuilt images from dockerhub, with our helper script:
```bash
./src/gso/harness/scripts/pull_images.sh
```

# 2. Running Evaluations


## 2.1 Preparing your submissions
Your agent/model/system's predictions should be a jsonl file with one line per task containing the following fields:
```python
{
    "instance_id": "str",         # gso task instance ID
    "model_patch": "str",         # generated patch to submit
    "model_name_or_path": "str",  # model name/path/identifier
}
```

## Evaluate your rollouts (Opt@K)

```bash
uv run src/gso/harness/opt_at_k.py \
    --model <modelname> \
    --prediction_paths <prediction_paths> \
    --timeout 3600 \
    --run_id <run_id> \
    --k 10 \
    --model <modelname>
```
- `--model` is the model/agent name to use for reporting.
- `--dataset_name` can be used for local jsonl file or a huggingface hub dataset.
- `--prediction_paths` is a space separated list of `K` predictions jsonl files (OR) a glob pattern.
- `--timeout` is the maximum time allowed for each task.
- `--run_id` is a unique identifier for your run.
- `--k` is the number of rollouts to evaluate.


> [!Note]
> Several helper scripts are in the [analysis/quantitative](../analysis/quantitative/) directory for plotting various results:
> - Opt@1 and Opt@K comparisons across models
> - Opt@K vs Steps vs Rollouts (performance across compute budgets)
> - Per-problem optimization stats
