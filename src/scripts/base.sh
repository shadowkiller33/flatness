proj_dir=/home/wtan12/flatness
export PYTHONPATH="/home/wtan12/flatness"
python $proj_dir/src/main.py \
    --mode max \
    --all_shots 8 \
    --models gpt2 \
    --bs 1 \
    --datasets agnews \
    --data-dir ${proj_dir}/data \
    --num_seeds 42 \
    --subsample_test_set 100