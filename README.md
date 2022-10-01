# flatness
This is a simpe implementation of the **Prompt Flatness**

```bash
python main.py \
    --mode max \
    --demo-shots 8 \
    --model gpt2 \
    --batch-size 16 \
    --dataset ag_news \
    --seed 42 \

```

* `mode`: the mode to calculate flatness
  * `average`: Standard fine-tuning
  * `max`: Prompt-based fine-tuning.
  * `average-max`: Prompt-based fine-tuning with demonstrations.
* `demo-shots`: Number of demonstrations
* `model`: the selected model
* `dataset`: dataset name
