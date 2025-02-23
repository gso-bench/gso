
# 1. Building PyPerf HF dataset

```bash
uv run src/pyperf/harness/build_dataset.py --exp_id pandas --push_to_hf --hf_username <hf_username>
```

<details>
<summary>The dataset created contains the following fields per task:</summary>

```json
{
    "instance_id": "str",               # pyperf task instance ID
    "repo": "str",                      # repository name
    "base_commit": "str",               # base commit hash
    "test_script": "str",               # test script to run
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



# 2. Building PyPerf Docker images

To push the dockers to dockerhub, you need to login first. Then run the following to build the images and push to dockerhub:
```bash
docker login

uv run src/pyperf/harness/prepare_images.py --dataset_name <dataset_name> --push_to_registry True --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```

- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--max_workers` can be used to scale up parallel builds



# 3. Running Evaluations

<details>
<summary>Prerequisite: You need to have all the docker images built and available.</summary>
<pre>./src/pyperf/harness/pull_images.sh -r slimshetty/pyperf-pandas -s</pre>
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

```bash
uv run src/pyperf/harness/run_evaluation.py --dataset_name <dataset_name> --predictions_path <predictions_path> --timeout 3600 --run_id <run_id>
```
- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--predictions_path` is the path to the predictions jsonl file.
- `--timeout` is the maximum time allowed for each task.
- `--run_id` is a unique identifier for the run.
