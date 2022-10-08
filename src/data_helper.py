from pathlib import Path
from datasets import load_dataset
from src.utils.dataset_utils import create_set, pashuffle, load_dataset, random_sampling
import logging
from copy import deepcopy
import numpy as np
import os

logger = logging.getLogger(__name__)
ag_news_prompts = [
    "What label best describes this news article?",
    "What is this a piece of news regarding for?",
    " What is the category of the following news?",
    "Which is the most relevant topic of the following news?",
    "Give the topic of the given text.",
    "Read the text below, provide its focused topic.",
    "Is this a piece of news regarding world, sport, business,or science?",
    "Which section of a newspaper would this article likely appear in?",
]


class DataHelper:
    def __init__(self, data_dir, dataset_name, k, seed) -> None:
        # download data to specified dir if not already exist
        self.data_path = os.path.join(data_dir, dataset_name)
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(f"{self.data_path}/train.csv"):
            raise ValueError(
                f"Download Dataset for {dataset_name} to {self.data_path} first!!"
            )

    def get_in_context_prompt(self, params, ins, seed=42, freeze_test_set=True):
        (
            all_train_sentences,
            all_train_labels,
            all_test_sentences,
            all_test_labels,
        ) = load_dataset(self.data_path, params, ins + "\n\n")
        # retrieve test set
        if params["subsample_test_set"] is None:
            # use all test
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0)  # always use seed 0 result if freeze
            else:
                np.random.seed(params["seed"])
            test_sentences, test_labels = random_sampling(
                all_test_sentences, all_test_labels, params["subsample_test_set"]
            )
            print(f"selecting {len(test_labels)} subsample of test set")
        # retrieve train set
        np.random.seed(seed)
        train_sentences, train_labels = random_sampling(
            all_train_sentences, all_train_labels, params["num_shots"]
        )
        return (train_sentences, train_labels, test_sentences, test_labels)

    @staticmethod
    def get_prompts():
        return ag_news_prompts

    @staticmethod
    def get_pertubed_set(prompt):
        out = set()
        while len(out) < 7:
            sss = pashuffle(prompt, perc=10)
            if sss != prompt:
                out.add(sss)
        return out
