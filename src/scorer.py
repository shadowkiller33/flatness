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

    def Flatness_correlation(self, flatness, performance, verbose=False):
        flatness = [float(i) / sum(flatness) for i in flatness]

        a = pearsonr(flatness, performance)[0]
        b = spearmanr(flatness, performance)[0]
        c = kendalltau(flatness, performance)[0]
        if verbose:
            print(
                f"The pearson correlation between {self.mode} flatness and acc is {a}"
            )
            print(
                f"The spearman correlation between {self.mode} flatness and acc is {b}"
            )
            print(
                f"The kendall correlation between {self.mode} flatness and acc is {c}"
            )
        return (a, b, c)

    def sen_correlation(self, sen, performance, verbose=False):
        sen = [float(i) / sum(sen) for i in sen]

        a = pearsonr(sen, performance)[0]
        b = spearmanr(sen, performance)[0]
        c = kendalltau(sen, performance)[0]
        if verbose:
            print(f"The pearson correlation between sensitivity and acc is {a}")
            print(f"The spearman correlation between sensitivity and acc is {b}")
            print(f"The kendall correlation between sensitivity and acc is {c}")
        return (a, b, c)

    def MI_correlation(self, MI, performance, verbose=False):
        MI = [float(i) / sum(MI) for i in MI]

        a = pearsonr(MI, performance)[0]
        b = spearmanr(MI, performance)[0]
        c = kendalltau(MI, performance)[0]
        if verbose:
            print(f"The pearson correlation between mutual information and acc is {a}")
            print(f"The spearman correlation between mutual information and acc is {b}")
            print(f"The kendall correlation between mutual information and acc is {c}")
        return (a, b, c)

    def ours_correlation(self, flat, performance, verbose=False):
        flat = [float(i) / sum(flat) for i in flat]

        a = pearsonr(flat, performance)[0]
        b = spearmanr(flat, performance)[0]
        c = kendalltau(flat, performance)[0]
        if verbose:
            print(f"The pearson correlation between flat and acc is {a}")
            print(f"The spearman correlation between flat and acc is {b}")
            print(f"The kendall correlation between flat and acc is {c}")
        return (a, b, c)

    def ours_correlation_MI(self, flatness, MI, performance, verbose=False):
        flatness = [float(i) / sum(flatness) for i in flatness]
        MI = [float(i) / sum(MI) for i in MI]
        performance = [float(i) / sum(performance) for i in performance]
        A, B, C = [], [], []
        for i in range(1000):
            result = [y + 0.01 * (i-500) * x for (x, y) in zip(flatness, MI)]
            A.append(pearsonr(result, performance)[0])
            B.append(spearmanr(result, performance)[0])
            C.append(kendalltau(result, performance)[0])
        index = A.index(max(A))
        if verbose:
            print(f"The best alpha (weighted factor) is {index}")
            print(f"The pearson correlation between ours (flatness + MI) and acc is {A[index]}")
            print(
                f"The spearman correlation between ours (flatness + MI) and acc is {B[index]}"
            )
            print(
                f"The kendall correlation between ours (flatness + MI) and acc is {C[index]}"
            )
        return (A[index], B[index], C[index])

    def ours_correlation_sen(self, flatness, sen, performance, verbose=False):
        flatness = [float(i) / sum(flatness) for i in flatness]
        sen = [float(i) / sum(sen) for i in sen]
        performance = [float(i) / sum(performance) for i in performance]
        A, B, C = [], [], []
        for i in range(1000):
            result = [y + 0.01 * (i-500) * x for (x, y) in zip(flatness, sen)]
            A.append(pearsonr(result, performance)[0])
            B.append(spearmanr(result, performance)[0])
            C.append(kendalltau(result, performance)[0])
        index = A.index(max(A))
        if verbose:
            print(f"The best alpha (weighted factor) is {index}")
            print(f"The pearson correlation between ours (flatness + sen) and acc is {A[index]}")
            print(
                f"The spearman correlation between ours (flatness + sen) and acc is {B[index]}"
            )
            print(
                f"The kendall correlation between ours (flatness + sen) and acc is {C[index]}"
            )
        return (A[index], B[index], C[index])

    def MI_sen_correlation(self, MI, sen, performance, verbose=False):
        A, B, C = [], [], []
        for i in range(1000):
            result = [y + 0.001 * (i-500)  * x for (x, y) in zip(MI, sen)]
            A.append(pearsonr(result, performance)[0])
            B.append(spearmanr(result, performance)[0])
            C.append(kendalltau(result, performance)[0])
        index = A.index(max(A))
        if verbose:
            print(f"The best alpha (weighted factor) is {index}")
            print(f"The pearson correlation between (MI + sen) and acc is {A[index]}")
            print(
                f"The spearman correlation between (MI + sen) and acc is {B[index]}"
            )
            print(
                f"The kendall correlation between (MI + sen) and acc is {C[index]}"
            )
        return (A[index], B[index], C[index])




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
