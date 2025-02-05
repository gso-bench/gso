
# Building PyPerf huggingface dataset

```bash
uv run src/pyperf/harness/build_dataset.py --exp_id pandas --push_to_hf --hf_username <hf_username>
```


# Building PyPerf Docker images

To push the dockers to dockerhub, you need to login to dockerhub first
```bash
docker login
```

Then run the following to build the docker images and push to dockerhub:
```bash
uv run src/pyperf/harness/prepare_images.py --dataset_name <dataset_name> --push_to_registry True --dockerhub_username <dockerhub_username> --dockerhub_repo <dockerhub_repo>
```

> --dataset_name can be a local jsonl file or a huggingface hub dataset
> --max_workers can be used to scale up parallel builds

