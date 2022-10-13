#!/bin/bash
#SBATCH -J gpt2xl_agnews
#SBATCH -A danielk_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=gpt2xl-ag-%x.%j.out
#SBATCH --error=gpt2xl-ag-%x.%j.err
#SBATCH --time=12:00:00

module load anaconda
conda activate flat
user_id=wtan12
proj_dir=/home/${user_id}/flatness
export PYTHONPATH="/home/${user_id}/flatness"
echo "start job"
srun python $proj_dir/src/main.py  \
    --model gpt2-xl --dataset agnews \
    --perturbed_num 7 --all_shots 4 \
    --bs 1 --num_seeds 3  --subsample_test_set 10 \
    --mode mean  --approx \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose
    
echo "end job"
