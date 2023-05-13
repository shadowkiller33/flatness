# Flatness-Aware Prompt Selection Improves Accuracy and Sample Efficiency

This is the official documentation for the paper **Flatness-Aware Prompt Selection Improves Accuracy and Sample Efficiency**. 

## Table of Contents

- [Installation](#installation)
- [Obtain scores](#Obtain scores)
- [Result format](#Result format)
- [Tune Alpha](#Tune Alpha)
- [Customization](#Customization)
- [Contact Us](#Contact Us)


## Installation
To run the codes, follow the steps below:
Install the required dependencies as followings:
```
pip install -r requirements.txt
```


## Obtain scores
Get the metrics scores for the prompts as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--model="gpt2" \
--dataset=agnews \
--num_seeds=1 \
--all_shots = 4 \
--subsample_test_set=512 \
--approx
```

* `all_shots`: Number of demonstrations
* `model`: the selected model
* `dataset`: dataset name
* `subsample_test_set`: size of test set to use to speed up eval. None means using all test set



## Result format
After running the codes above, you'll get results (pickle file).
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
      perf: performance (acc),

    }
    // seed-level info: correlations across prompt
    sen_p:  ,
    sen_s: ,
    mi_p: ..,
    mi_s: ..,
  }
  // top level info like avg sensitivity avg accuracy etc. is calculated by print_results function. they are not stored in the pickle
}
```
* `id`: the prompt id
* `promt`: the contents of prompt
* `sen`: the sensitivity of the prompt
* `mi`: multual information of the prompt
* `perf`: accuracy of the prompt
* `sen_p`: Pearson correlation between performance and sensitivity
* `sen_s`: Spearman correlation between performance and sensitivity
* `mi_p`: Pearson correlation between performance and mutual information
* `mi_s`: Spearman correlation between performance and mutual information


## Tune Alpha
After obtaining the correlation between metrics scores and performance on the dev-set, we tune the alpha that maximizes the correlation or other metrics (e.g., NDCG). Then fix it, and run on the large test set.

## Customization
To set your own custom prompts, you can change it at promptset in [main.py](https://github.com/shadowkiller33/flatness/blob/main/main.py)



## Contact Us
If you have any questions, suggestions, or concerns, please reach out to us.
