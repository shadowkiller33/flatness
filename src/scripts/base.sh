#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --output=ag-%x.%j.out
#SBATCH --time=12:00:00
module load anaconda
conda info --envs
source activate flat


user_id=lshen30
proj_dir=/data/danielk/${user_id}/flat
export PYTHONPATH="/data/danielk/${user_id}/flat"
python -u $proj_dir/src/main.py  \
    --model gpt2  --dataset agnews \
    --perturbed_num 7 --all_shots 5 \
    --bs 20 --num_seeds 2  --subsample_test_set 128 \
    --mode mean  --approx \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose


