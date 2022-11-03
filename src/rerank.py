from curses.ascii import isdigit
import os
import pickle
from src.scorer import Scorer
from src.utils.io_utils import print_results
import numpy as np


def find_best_alpha(file):
    with open(file, "rb") as f:
        result_table = pickle.load(f)
    scorer = Scorer("mean", 1, None)
    verbose = False

    for seed_id in result_table.keys():  # should only have 1 seed
        sen_list, mi_list, flat_list, perf_list = [], [], [], []
        for prompt_id in result_table[seed_id].keys():
            if isinstance(prompt_id, int):
                sen_list.append(result_table[seed_id][prompt_id]["sen"])
                mi_list.append(result_table[seed_id][prompt_id]["mi"])
                flat_list.append(result_table[seed_id][prompt_id]["flat"])
                perf_list.append(result_table[seed_id][prompt_id]["perf"])

    # Flat + MI
    flat_mi_alpha = scorer.ours_correlation_MI(
        mi_list, flat_list, perf_list, verbose=verbose, return_alpha=True
    )

    # Flat + sensitivity
    flat_sen_alpha = scorer.ours_correlation_sen(
        sen_list, flat_list, perf_list, verbose=verbose, return_alpha=True
    )

    # MI + sensitivity
    mi_sen_alpha = scorer.MI_sen_correlation(
        sen_list, mi_list, perf_list, verbose=verbose, return_alpha=True
    )
    return (flat_mi_alpha, flat_sen_alpha, mi_sen_alpha)


def find_rerank(gold, inp):
    out = [0 for _ in range(20)]
    pred = inp[gold]
    inp_reorder = np.sort(inp)[::-1]
    index = np.where(inp_reorder == pred)[0][0]
    out[index] = 1
    return out


def find_rerank_comb(gold, inp1, inp2, alpha):
    out = [0 for _ in range(20)]
    inp = inp1 * (alpha - 500) * 0.0002 + inp2
    pred = inp[gold]
    inp_reorder = np.sort(inp)[::-1]
    index = np.where(inp_reorder == pred)[0][0]
    out[index] = 1
    return out


def rerank_with_alpha(file, flat_mi_alpha, flat_sen_alpha, mi_sen_alpha):
    with open(file, "rb") as f:
        result_table = pickle.load(f)
    scorer = Scorer("mean", 1, None)
    verbose = False

    sen_rs, mi_rs, flat_rs = [], [], []
    sen_mi_rs, flat_mi_rs, flat_sen_rs = [], [], []
    for seed_id in result_table.keys():
        sen_list, mi_list, flat_list, perf_list = [], [], [], []
        for prompt_id in result_table[seed_id].keys():
            if isinstance(prompt_id, int):
                sen_list.append(result_table[seed_id][prompt_id]["sen"])
                mi_list.append(result_table[seed_id][prompt_id]["mi"])
                flat_list.append(result_table[seed_id][prompt_id]["flat"])
                perf_list.append(result_table[seed_id][prompt_id]["perf"])
        sen_list = np.array(sen_list)
        mi_list = np.array(mi_list)
        flat_list = np.array(flat_list)
        perf_list = np.array(perf_list)
        # compute ranking and top@1
        gold = np.argmax(perf_list)
        sen_rerank = find_rerank(gold, sen_list)
        mi_rerank = find_rerank(gold, mi_list)
        flat_rerank = find_rerank(gold, flat_list)
        sen_mi_rerank = find_rerank_comb(gold, mi_list, sen_list, mi_sen_alpha)
        flat_sen_rerank = find_rerank_comb(gold, flat_list, sen_list, flat_sen_alpha)
        flat_mi_rerank = find_rerank_comb(gold, flat_list, mi_list, flat_mi_alpha)

        sen_rs.append(sen_rerank)
        mi_rs.append(mi_rerank)
        flat_rs.append(flat_rerank)
        sen_mi_rs.append(sen_mi_rerank)
        flat_mi_rs.append(flat_mi_rerank)
        flat_sen_rs.append(flat_sen_rerank)

    print(f"SEN Rerank: {mean_reciprocal_rank(sen_rs):.3}")
    print(f"MI Rerank: {mean_reciprocal_rank(mi_rs):.3}")
    print(f"Flat Rerank: {mean_reciprocal_rank(flat_rs):.3}")
    print(f"SEN+MI Rerank: {mean_reciprocal_rank(sen_mi_rs):.3}")
    print(f"Flat+MI Rerank: {mean_reciprocal_rank(flat_mi_rs):.3}")
    print(f"Flat+Sen Rerank: {mean_reciprocal_rank(flat_sen_rs):.3}")


def get_file(dataset, model_type):
    path = "/home/wtan12/flatness/output"
    test_file, rerank_file = None, None
    for _, _, files in os.walk(path):
        for file in files:
            # find related dataset and model file
            if file.startswith(f"{dataset}_gpt2{model_type}"):
                if "rerank" in file:
                    rerank_file = f"{path}/{file}"
                else:
                    test_file = f"{path}/{file}"
            if rerank_file and test_file:
                return (test_file, rerank_file)
        break
    raise FileNotFoundError(f"missing pickle file for {dataset}, {model_type}")


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                    (first element is the first item)
    Returns:
            Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return 100 * np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def top1_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 if r.size and r[0] == 0 else 0.0 for r in rs])


if __name__ == "__main__":
    dataset = "dbpedia"  # change to other dataset
    for model_type in ["_", "-medium", "-large", "-xl"]:
        (test_file, rerank_file) = get_file(dataset, model_type)
        print(f"evaluating {test_file}")
        # infer best alpha from rerank file (dev set)
        (flat_mi_alpha, flat_sen_alpha, mi_sen_alpha) = find_best_alpha(rerank_file)
        # evaluate rerank acc from rerank results
        rerank_with_alpha(test_file, flat_mi_alpha, flat_sen_alpha, mi_sen_alpha)
