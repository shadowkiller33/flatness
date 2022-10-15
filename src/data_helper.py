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
    "What is the category of the following news?",
    "Which is the most relevant topic of the following news?",
    "Give the topic of the given text.",
    "Read the text below, provide its focused topic.",
    "Is this a piece of news regarding world, sport, business,or science?",
    "Which section of a newspaper would this article likely appear in?",
    "What label would you use to characterize this news item?",
    "What term best sums up this news report?",
    "Which category most accurately sums up this news item?",
    "What label would you use to characterize this news story?",
    "Is this news related to the world, sports, business, or science?",
    "Does this news story have anything to do with the world, sports, business, or science?",
    "Read the paragraph below and explain its specific subject.",
    "Please read the following material and explain its main point.",
    "Provide your thoughts on the content below after reading it.",
    "Describe the text's subject as follows.",
    "For what purpose does this news item exist?",
    "Are there any world-related, sports, business, or science-related stories in this news?",
]


class DataHelper:
    def __init__(self, data_dir, dataset_name) -> None:
        # download data to specified dir if not already exist
        self.data_path = os.path.join(data_dir, dataset_name)
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(f"{self.data_path}/train.csv"):
            raise ValueError(
                f"Download Dataset for {dataset_name} to {self.data_path} first!!"
            )

    def get_in_context_prompt(
        self, params, ins, seed, freeze_test_set=True, verbose=False
    ):
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
            if verbose:
                print(
                    f"selecting full test set ({len(all_test_labels)} examples)",
                )
        else:
            if freeze_test_set:
                np.random.seed(0)  # always use seed 0 result if freeze
            else:
                np.random.seed(params["seed"])
            test_sentences, test_labels = random_sampling(
                all_test_sentences, all_test_labels, params["subsample_test_set"]
            )
            if verbose:
                print(f"selecting {len(test_labels)} subsample of test set")
        # retrieve train set
        np.random.seed(0)
        train_sentences, train_labels = random_sampling(
            all_train_sentences, all_train_labels, params["num_shots"]
        )
        return (train_sentences, train_labels, test_sentences, test_labels)

    @staticmethod
    def get_prompts(dataset):
        if dataset == "agnews":
            return ag_news_prompts
        raise ValueError("dataset name not recognized")

    @staticmethod
    def get_pertubed_set(prompt, num=7):
        out = set()
        while len(out) < num:
            sss = pashuffle(prompt)
            if sss != prompt:
                out.add(sss)
        return out

    @staticmethod
    def get_prompt_order(train_sentences, train_labels, num=7):
        import random

        order_list = []
        c = list(zip(train_sentences, train_labels))
        for i in range(num):
            random.shuffle(c)
            train_sentences, train_labels = zip(*c)
            if (train_sentences, train_labels) not in order_list:
                order_list.append((train_sentences, train_labels))

        return order_list
        