# flatness
This is a simpe implementation of the **Prompt Flatness**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
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
* `subsample_test_set`: size of test set to use to speed up eval. None means using all test set

To set your own custom prompts, you can change it at promptset in [main.py](https://github.com/shadowkiller33/flatness/blob/main/main.py)

## Result Tree Printout Format
For each experiment, we store a result tree in the following format:
```
{
  seed_id: {
    id: {
      // prompt level info
      id: prompt_id,
      promt: prompt_text,
      sen: sen_score,
      mi: mi_score,
      perf: performance_score,

      acc: original_acc,
      acc_c: calibrated_acc,
      p_cf: [context_free_probs],
      others: train/test_sentences, etc.
    }
    // seed-level info: correlations across prompt
    sen_p: sen_pearson_corr,
    sen_s: sen_spearman_corr,
    sen_k: sen_kendalltau_corr,
    mi_p: ..,
    mi_s: ..,
    mi_k: ..,
    ours_p: ..,
    ours_s: ..,
    ours_k: ..,
  }
  // top level info like avg sensitivity avg accuracy etc. is calculated by print_results function. they are not stored in the pickle
}
```
for different models and datasets, they are serialized in different location, so there is no need to store that information in result tree.
