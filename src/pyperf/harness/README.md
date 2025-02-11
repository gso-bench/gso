
# Building PyPerf huggingface dataset

```bash
uv run src/pyperf/harness/build_dataset.py --exp_id pandas --push_to_hf --hf_username <hf_username>
```

You can use the local dataset or the huggingface dataset to solve tasks in the PyPerf benchmark. The dataset contains the following fields per task:
```json
{
    TODO: complete this
}
```


# Building PyPerf Docker images

To push the dockers to dockerhub, you need to login first. Then run the following to build the docker images and push to dockerhub:
```bash
docker login

uv run src/pyperf/harness/prepare_images.py --dataset_name <dataset_name> --push_to_registry True --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```

- `--dataset_name` can be a local jsonl file or a huggingface hub dataset.
- `--max_workers` can be used to scale up parallel builds



# Running PyPerf Evaluations

<details>
<summary>Prerequisite: You need to have all the docker images built and available.</summary>
```bash
./src/pyperf/harness/pull_images.sh -r slimshetty/pyperf-pandas -s
```
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
