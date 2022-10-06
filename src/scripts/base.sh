#!/bin/bash
#SBATCH -A danielk_gpu​
#SBATCH --partition=a100​
#SBATCH --gres=gpu:1​
#SBATCH -N 1​
#SBATCH --ntasks-per-node=12​


source /etc/profile.d/modules.sh
module load python

python main.py --model gpt2   --dataset agnews   --all_shots 8 --num_seeds 1   --mode mean  --approx --data-dir /home/lshen30/flat/data
#### execute code and write output file to OUT-24log.
#time mpiexec ./code-mvapich.x > OUT-24log
echo "Finished with job $SLURM_JOBID"

#### mpiexec by default launches number of tasks requested