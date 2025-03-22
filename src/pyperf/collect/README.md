# Building PyPerf HF dataset

> The tasks in this dataset are built from the results of pyperf pipeline.

```bash
uv run src/pyperf/collect/build_dataset.py --exp_id pandas --push_to_hf --hf_username <hf_username>
```

The dataset will contain the following fields per task:
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