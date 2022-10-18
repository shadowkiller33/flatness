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

cb_prompt = [
    "Suppose we have the following premise, Can we infer that hypothesis? Yes, no, or maybe?",
    "Based on the previous premise, is it true for the hypothesis?",
    "See on the following information, is the claim right?",
    "Given that premise, does it follow that hypothesis? Yes, no, or maybe?",
    "Given the premise, are we justified in saying that hypothesis? Yes, no, or maybe?",
    "Based on the text, question: hypothesis is True, False, or Neither?",
    "Keeping in mind the above text, consider: hypothesis  is always, sometimes, or never correct?"
    "Given premise. Is it guaranteed true that hypothesis? Yes, no, or maybe?",
    "Given that premise. Therefore, it must be true that hypothesis? Yes, no, or maybe?",
    "Assume it is true that premise. Therefore, hypothesis is guaranteed, possible, or impossible?",
    "Using only the following description and what you know about the world, hypothesis is definitely correct, incorrect, or inconclusive?",
    "Take the following as truth. Then the hypothesis is true, false, or inconclusive?",
    "Can we derive that hypothesis if we have the following premise? Yes, no, or perhaps?",
    "Can we arrive at that conclusion if we possess the following information? Possibly, no, or both?",
    "Does that premise flow from the given premise? Yes, no, or perhaps?",
    "Does that information support the claim?",
    "Is the assertion accurate in light of such information?",
    "Considering the text, which of the following statements is True, False, or Both?",
    "Think about the question: Is hypothesis always, occasionally, or never correct?",
    "Can we derive that conclusion if we have the following information? Yes, no, or possibly?",
]

dbpedia_prompt = [
    "What label best describes this paragraph?",
    "What is this paragraph regarding for?",
    "What is the category of the following paragraph?",
    "Which is the most relevant topic of the following paragraph?",
    "Give the topic of the given text.",
    "Read the text below, provide its focused topic.",
    "Is this paragraph regarding company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work?",
    "What label would you use to characterize this paragraph?",
    "What term best sums up this paragraph?",
    "Which category most accurately sums up this paragraph?",
    "What label would you use to characterize this paragraph?",
    "Is this paragraph related to company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work?",
    "Does this news story have anything to do with company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work?",
    "Read the paragraph below and explain its specific subject.",
    "Please read the following material and explain its main point.",
    "Describe the text's subject as follows.",
    "Are there any company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work content in this paragraph?"
    "Given a list of categories: company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work, what category does the paragraph belong to?",
    "Pick one category for the following text. The options are - company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work",
    "Given a choice of categories company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work, the text refers to which one? ",
]

sst2_prompt = [
    "What sentiment best describes this review?",
    "Which emotion best sums up this review?" "Is this review positive or negative?",
    "Give the sentiment of the given text.",
    "Read the text below, does the review has positive or negative sentiment?",
    "Read the material below; is there a positive or negative tone to the review?",
    "What sentiment would you use to characterize this review?",
    "Is this a piece of postive or negative review?",
    "Provide your thoughts on the content below after reading it.",
    "Are there positive or negative sentiments in this review?"
    "What is the opinion of this review? Positive or negative?",
    "What is the sentiment of the following review?",
    "Which is the most relevant sentiment of the following review? Positive or negative?",
    "Read the text below, provide its sentiment.",
    "What term best sums up this review?",
    "Which sentiment most accurately sums up this review?",
    "What label would you use to characterize this review?",
    "Please read the following material and explain its main point.",
    "Provide your thoughts on the content below after reading it.",
    "Describe the text's subject as follows.",
    "How do you feel about the following review? Positive or negative?",
]


class DataHelper:
    def __init__(self, data_dir, dataset_name) -> None:
        # download data to specified dir if not already exist
        self.data_path = os.path.join(data_dir, dataset_name)

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
                print(f"selecting full test set ({len(all_test_labels)} examples)")
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
        elif dataset == "cb":
            return cb_prompt
        elif dataset == "dbpedia":
            return dbpedia_prompt
        elif dataset == "sst2":
            return sst2_prompt
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
