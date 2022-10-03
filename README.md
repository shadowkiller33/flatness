# flatness
This is a simpe implementation of the **Prompt Flatness**

```bash
CUDA_VISIBLE_DEVICES=0 python run_classification.py \
--model="gpt2" \
--dataset=agnews \
--num_seeds=1 \
--mode = 'mean'
--all_shots = 4 \
--subsample_test_set=300 \
--approx
```

* `mode`: the mode to calculate flatness
  * `mean`: L1 distance
  * `max`: Maximum of difference (L1)
  * `mean-max`: combine
* `all_shots`: Number of demonstrations
* `model`: the selected model
* `dataset`: dataset name
