#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --output=/home/wtan12/flatness/output/slurm/agnews-gpt.out
#SBATCH --time=72:00:00

module load anaconda
conda activate flat
user_id=wtan12
proj_dir=/home/${user_id}/flatness
export PYTHONPATH="/home/${user_id}/flatness"
python -u $proj_dir/src/main.py  \
    --model gpt2 --dataset agnews \
    --mode mean \
    --perturbed_num 7 --all_shots 5 --approx \
    --bs 2 --num_seeds 3  --subsample_test_set 512 \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose


