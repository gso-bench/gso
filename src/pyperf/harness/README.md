<details>
<summary>Prerequisite: A pyperf dataset that must contain the following fields per task:</summary>

```python
{
    "instance_id": "str",               # pyperf task instance ID
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


# 1. Building PyPerf Docker images

To push the dockers to dockerhub, you need to login first. Then run the following to build the images and push to dockerhub:
```bash
docker login

uv run src/pyperf/harness/prepare_images.py --dataset_name <dataset_name> --push_to_registry True --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```

- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--max_workers` can be used to scale up parallel builds



# 2. Running Evaluations

<details>
<summary>Prerequisite: You need to have all the docker images built and available.</summary>
<pre>./src/pyperf/harness/scripts/pull_images.sh -r slimshetty/pyperf-pandas -s</pre>
TODO: pull images as and when needed
</details></br>

Your system's predictions should be a jsonl file with one line per task containing the following fields:
```python
{
    "instance_id": "str",         # pyperf task instance ID
    "model_patch": "str",         # generated patch file to submit
    "model_name_or_path": "str",  # model name/path/identifier
}
```

## 2.1 Evaluate a single rollout (Opt@1)

```bash
uv run src/pyperf/harness/run_evaluation.py --dataset_name <dataset_name> --predictions_path <predictions_path> --timeout 3600 --run_id <run_id>
```
- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--predictions_path` is the path to the predictions jsonl file.
- `--timeout` is the maximum time allowed for each task.
- `--run_id` is a unique identifier for the run.


## 2.2 Evaluate multiple rollouts (Opt@K)

```bash
uv run src/pyperf/harness/opt@k.py --dataset_name <dataset_name> --prediction_paths <prediction_paths> --timeout 3600 --run_id <run_id> --k 10 --model <modelname>
```
- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--prediction_paths` is a space separated list of predictions jsonl files (OR) a glob pattern.
- `--timeout` is the maximum time allowed for each task.
- `--run_id` is a unique identifier for the run.
- `--k` is the number of rollouts to evaluate.
- `--model` is the model/agent name to use for reporting.

> [!Note]
> Find scripts to plot results such as Opt@K and Speedups acheived in ([/scripts](./scripts/))
