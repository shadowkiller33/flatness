# script to use with tmux session requested by
# salloc -J session -A danielk_gpu  --nodes=1 --ntasks-per-node=1 --mem-per-cpu=4GB --time=60:00 --partition=a100 --gpus-per-node=1 --cpus-per-task=4 srun --pty bash
module load anaconda
conda activate flat
user_id=wtan12
proj_dir=/home/${user_id}/flatness
export PYTHONPATH="/home/${user_id}/flatness"
python $proj_dir/src/main.py  \
    --model gpt2-xl  --dataset agnews \
    --perturbed_num 7 --all_shots 4 \
    --bs 4 --num_seeds 1  --subsample_test_set 20 --approx \
    --data-dir $proj_dir/data \
    --output-dir $proj_dir/output \
    --verbose --use-submit
