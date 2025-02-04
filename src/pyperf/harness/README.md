
# Building PyPerf huggingface dataset

```bash
uv run src/pyperf/harness/build_dataset.py --exp_id pandas --push_to_hf --hf_username <hf_username>
```


# Building PyPerf Docker images

To push the dockers to dockerhub, you need to login to dockerhub first
```bash
docker login
```

> From a local jsonl dataset
```bash
uv run src/pyperf/harness/prepare_images.py --dataset_name pyperf_dataset.jsonl
--push_to_registry --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```

> From a huggingface hub dataset
```bash
uv run src/pyperf/harness/prepare_images.py --dataset_name manishs/pyperf_dataset --push_to_registry --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```


