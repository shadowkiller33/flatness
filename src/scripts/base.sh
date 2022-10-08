#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --output=gpt2xl-ag-%x.%j.out
#SBATCH --time=12:00:00
module load anaconda
conda info --envs
source activate flat


proj_dir=/home/lshen30/flat
export PYTHONPATH="/home/lshen30/flat"
python $proj_dir/src/main.py  --model gpt2-xl  --dataset agnews  --all_shots 4  --bs 12 --num_seeds 1  --subsample_test_set 500  --mode mean  --approx  --data-dir /home/lshen30/flat/data
