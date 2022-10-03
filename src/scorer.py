from src.utils.dataset_utils import loader_labeled
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr, kendalltau
import logging
import torch
from functools import reduce
import numpy as np

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
            # normalization = reduce(lambda x, y: x*y, list(avg_tensor.size()))
            # mean = torch.sum(avg_tensor)/normalization
            return avg_tensor.detach().cpu().item()

        elif self.mode == "max":
            max_tensor = torch.max(torch.stack(input))
            return max_tensor.detach().cpu().item()

        elif self.mode == "avg_max":
            avg_tensor = torch.mean(torch.stack(input))
            max_tensor = torch.max(avg_tensor)
            return max_tensor.detach().cpu().item()

    def eval_accuracy(self, all_label_probs, test_labels, mode=None, p_cf=None):
        # evaluate the accuracy with and without contextual calibration
        num_classes = all_label_probs.shape[1]
        if p_cf is None:
            # do not calibrate
            W = np.identity(num_classes)
            b = np.zeros([num_classes, 1])
        else:
            # calibrate
            if mode == "diagonal_W":
                W = np.linalg.inv(np.identity(num_classes) * p_cf)
                b = np.zeros([num_classes, 1])
            elif mode == "identity_W":
                W = np.identity(num_classes)
                b = -1 * np.expand_dims(p_cf, axis=-1)
            else:
                assert False

        correctness_list = []
        assert len(all_label_probs) == len(test_labels)
        for label_probs, true_label in zip(all_label_probs, test_labels):
            label_probs = label_probs / np.sum(label_probs)  # normalize to 1

            calibrate_label_probs = (
                np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
            )

            ans_label = np.argmax(calibrate_label_probs)
            if ans_label == true_label:
                correctness_list.append(1)
            else:
                correctness_list.append(0)
        return np.mean(correctness_list)
