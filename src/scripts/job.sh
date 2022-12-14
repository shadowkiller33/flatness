#!/bin/bash

module load anaconda
conda activate flat
user_id=wtan12
proj_dir=/home/${user_id}/flatness
export PYTHONPATH="/home/${user_id}/flatness"
python -u $proj_dir/src/main.py  \
    --model gpt2-xl --dataset agnews \
    --perturbed_num 7 --all_shots 5 --approx \
    --bs 4 --num_seeds 3  --subsample_test_set 512 \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose

