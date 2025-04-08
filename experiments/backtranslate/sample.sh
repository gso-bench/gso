#!/bin/bash
# set -e

echo "========= Sampling plans for backtranslation ========="
echo ""
uv run experiments/backtranslate/sample_plans.py --model-name o3-mini --max_tokens 24000 --n 5 --use_cache False --temperature 0.5 --multiprocess 50
echo "==============================================================="
