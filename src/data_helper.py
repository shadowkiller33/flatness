from datasets import load_dataset
from src.utils.dataset_utils import create_set, pashuffle
import logging
from copy import deepcopy

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
    def __init__(self, dataset_name, k, seed) -> None:
        self.datasets = load_dataset(dataset_name)
        self.label_dict = {"0": "world", "1": "sports", "2": "business", "3": "science"}

        (
            self.demo_labels,
            self.demo_inputs,
            self.test_inputs,
            self.test_labels,
        ) = create_set(self.datasets, k, seed, self.label_dict)
        print(
            f"Sample Demo: Input: {self.demo_inputs[0]}, Output: {self.demo_labels[0]}"
        )

    def get_in_context_prompt(self, ins):
        for x, y in zip(self.demo_labels, self.demo_inputs):
            ins += "\n" + "Input:" + y + "\n" + "Output:" + self.label_dict[str(x)]
        prompts = deepcopy(self.test_inputs)
        for i in range(len(self.test_inputs)):
            prompts[i] = ins + "\n" + "Input:" + self.test_inputs[i] + "\n" + "Output:"
        return (prompts, self.test_labels)

    @staticmethod
    def get_prompts():
        return ag_news_prompts

    @staticmethod
    def get_pertubed_set(prompt):
        out = set()
        while len(out) < 10:
            sss = pashuffle(prompt, perc=10)
            if sss + "\n" != prompt:
                out.add(sss)
        return out
