proj_dir=/home/wtan12/flatness
export PYTHONPATH="/home/wtan12/flatness"
python $proj_dir/src/main.py
    --mode max \
    --demo-shots 8 \
    --model gpt2 \
    --batch-size 16 \
    --dataset ag_news \
    --seed 42 \