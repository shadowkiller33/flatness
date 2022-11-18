#!/bin/bash

# gpt2, gpt2-medium, gpt2-large, gpt2-xl
module load anaconda
conda activate flat
user_id=wtan12
proj_dir=/home/${user_id}/flatness
export PYTHONPATH="/home/${user_id}/flatness"
python -u $proj_dir/src/abstain.py  \
    --model gpt2 --dataset agnews --mode mean \
    --perturbed_num 7 --all_shots 5 --approx \
    --bs 4 --num_seeds 1  --subsample_test_set 32 \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose

