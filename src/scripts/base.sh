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
proj_dir=/home/${user_id}/flat
export PYTHONPATH="/home/${user_id}/flat"
python -u $proj_dir/src/main.py  \
    --model gpt2  --dataset agnews \
    --perturbed_num 7 --all_shots 4 \
    --bs 16 --num_seeds 3  --subsample_test_set 50 \
    --mode mean  --approx \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose


