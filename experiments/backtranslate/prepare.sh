#!/bin/bash
# set -e

echo "========= Preparing dataset for backtranslation ========="
echo ""
uv run experiments/backtranslate/prepare_dataset.py --exp_id oversample --push_to_hf --hf_username manishs --dataset_name pyperf-exteneded
echo "==============================================================="


echo "========= Sampling plans for backtranslation ========="
echo ""
uv run experiments/backtranslate/sample_plans.py --model-name o3-mini --max_tokens 24000 --n 2 --use_cache False --temperature 0.5
echo "==============================================================="
