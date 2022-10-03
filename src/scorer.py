from src.utils.dataset_utils import loader_labeled
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr, kendalltau
import logging
import torch
from functools import reduce

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, mode, batch_size, tokenizer) -> None:
        self.mode = mode
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def score_acc(self, generator, test_inputs, test_labels):
        # FIXME: max-seq-len is hard coded for now, and :20 should be removed
        test_dataset = loader_labeled(
            test_inputs[:20],
            test_labels[:20],
            self.tokenizer,
            1000,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=False
        )
        acc_total = generator.validate(test_loader)
        return acc_total

    def correlation(self, flatness, performance):
        a = pearsonr(flatness, performance)[0]
        b = spearmanr(flatness, performance)[0]
        c = kendalltau(flatness, performance)[0]
        print(
            f"The pearson correlation between {self.mode} flatness and human score is {a}"
        )
        print(
            f"The spearman correlation between {self.mode} flatness and human score is {b}"
        )
        print(
            f"The kendall correlation between {self.mode} flatness and human score is {c}"
        )

    def flat(self, input):
        if self.mode == "mean":
            avg_tensor = torch.mean(torch.stack(input))
            normalization = reduce(lambda x, y: x * y, list(avg_tensor.size()))
            mean = torch.sum(avg_tensor) / normalization
            return mean.detach().cpu().item()

        elif self.mode == "max":
            max_tensor = torch.max(torch.stack(input))
            return max_tensor.detach().cpu().item()

        elif self.mode == "avg_max":
            avg_tensor = torch.mean(torch.stack(input))
            max_tensor = torch.max(avg_tensor)
            return max_tensor.detach().cpu().item()
