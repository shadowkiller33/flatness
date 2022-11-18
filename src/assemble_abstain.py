from curses.ascii import isdigit
import os
import pickle
import numpy as np
import math
from collections import defaultdict

dataset2label = {"agnews": 4, "dbpedia": 14, "sst2": 2}


def f1(hyp, target, dataset):
    n = len(hyp)
    assert n == len(target)
    category = dataset2label[dataset]  # change this number for different task
    confusion_matrix = np.array([[0 for _ in range(category)] for _ in range(category)])
    for i in range(n):
        h = hyp[i]
        t = target[i]
        confusion_matrix[t, h] += 1
    f1s = []
    for c in range(category):
        # compute TP, FP, FN for each class
        tp = confusion_matrix[c][c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp
        # special case:
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(1)
        elif tp == 0:
            f1s.append(0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            class_f1 = 2 * precision * recall / (precision + recall)
            f1s.append(class_f1)
    return sum(f1s) / len(f1s)


def normalize(x):
    min_x, max_x = min(x), max(x)
    for i in range(len(x)):
        x[i] = (x[i] - min_x) / max_x
    return x


def select_pred(coverage, confidence, yhat, ytrue):
    total = len(confidence)
    select_amount = math.floor(coverage * total)
    out1, out2 = [], []
    out_true1, out_true2 = [], []
    conf_tuple = []
    for i in range(total):
        conf_tuple.append((i, confidence[i]))
    sort_conf1 = sorted(conf_tuple, key=lambda x: x[1], reverse=True)
    sort_conf2 = sorted(conf_tuple, key=lambda x: x[1], reverse=False)
    for k in range(select_amount):
        index = sort_conf1[k][0]
        out1.append(yhat[index])
        out_true1.append(ytrue[index])
        index = sort_conf2[k][0]
        out2.append(yhat[index])
        out_true2.append(ytrue[index])
    return out1, out_true1, out2, out_true2


def update_result_dict(table, key, coverage, value):
    if key in table[coverage]:
        table[coverage][key].append(value)
    else:
        table[coverage][key] = [value]
    return table


def choose_alpha(coverage, yhat, ytrue, sen, flat, dataset):
    sen = np.array(sen)
    flat = np.array(flat)
    maxf1 = -float("inf")
    best_alpha = None
    for i in range(1000):
        alpha = 0.0002 * (i - 500)
        senflat = sen + alpha * flat
        sf_hat, sf_true, sf_hat1, sf_true1 = select_pred(coverage, senflat, yhat, ytrue)
        sf_f1 = max(f1(sf_hat, sf_true, dataset), f1(sf_hat1, sf_true1, dataset))
        if sf_f1 > maxf1:
            maxf1 = sf_f1
            best_alpha = alpha

    return best_alpha, maxf1


def main(result_table, dataset):
    f1_results = defaultdict(dict)
    for seed_id in result_table.keys():
        for prompt_id in result_table[seed_id].keys():
            maxprob = np.array(result_table[seed_id][prompt_id]["maxprob"]).tolist()
            sen = np.array(result_table[seed_id][prompt_id]["sen"]).tolist()
            flat = np.array(result_table[seed_id][prompt_id]["flat"]).tolist()
            yhat = result_table[seed_id][prompt_id]["yhat"]
            ytrue = result_table[seed_id][prompt_id]["ytrue"]

            # create devset for choosing alpha
            devsize = math.floor(len(yhat) * 0.2)
            dev_yhat, dev_ytrue = yhat[:devsize], ytrue[:devsize]
            test_yhat, test_ytrue = yhat[devsize:], ytrue[devsize:]

            # normalize to 0-1 region
            sen = normalize(sen)
            flat = normalize(flat)
            maxprob = normalize(maxprob)

            # f1_no_abstain = f1(yhat, ytrue, dataset)
            # print(f"F1 score with 100 coverage: {f1_no_abstain}")

            for coverage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                # print(f"Compute F1 with coverage {coverage}")

                sen_yhat, sen_y, sen_yhat1, sen_y1 = select_pred(
                    coverage, sen[devsize:], test_yhat, test_ytrue
                )
                sen_f1 = max(
                    f1(sen_yhat, sen_y, dataset), f1(sen_yhat1, sen_y1, dataset)
                )

                f1_results = update_result_dict(f1_results, "sen", coverage, sen_f1)

                # print(f"SEN F1: {sen_f1}")

                flat_yhat, flat_y, flat_yhat1, flat_y1 = select_pred(
                    coverage, flat[devsize:], test_yhat, test_ytrue
                )
                flat_f1 = max(
                    f1(flat_yhat, flat_y, dataset), f1(flat_yhat1, flat_y1, dataset)
                )
                f1_results = update_result_dict(f1_results, "flat", coverage, flat_f1)
                # print(f"Flat F1: {flat_f1}")

                maxprob_yhat, maxprob_y, maxprob_yhat1, maxprob_y1 = select_pred(
                    coverage, maxprob[devsize:], test_yhat, test_ytrue
                )
                maxprob_f1 = max(
                    f1(maxprob_yhat, maxprob_y, dataset),
                    f1(maxprob_yhat1, maxprob_y1, dataset),
                )
                f1_results = update_result_dict(
                    f1_results, "maxprob", coverage, maxprob_f1
                )
                # print(f"Maxprob F1: {maxprob_f1}")

                alpha, maxf1 = choose_alpha(
                    coverage,
                    dev_yhat,
                    dev_ytrue,
                    sen[:devsize],
                    flat[:devsize],
                    dataset,
                )
                senflat = np.array(sen) + np.array(flat) * alpha
                senflat = senflat.tolist()[devsize:]

                sen_flat_yhat, sen_flat_y, sen_flat_yhat1, sen_flat_y1 = select_pred(
                    coverage, senflat, test_yhat, test_ytrue
                )
                sen_flat_f1 = max(
                    f1(sen_flat_yhat, sen_flat_y, dataset),
                    f1(sen_flat_yhat1, sen_flat_y1, dataset),
                )
                f1_results = update_result_dict(
                    f1_results, "sen_flat", coverage, sen_flat_f1
                )

        print("Coverage\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t0.7\t0.8\t0.9\t1")
        sen_result = []
        flat_result = []
        maxprob_result = []
        sen_flat_result = []
        for k in f1_results.keys():
            table = f1_results[k]
            sen_avg = 100 * sum(table["sen"]) / len(table["sen"])
            flat_avg = 100 * sum(table["flat"]) / len(table["flat"])
            maxprob_avg = 100 * sum(table["maxprob"]) / len(table["maxprob"])
            sen_flat_avg = 100 * sum(table["sen_flat"]) / len(table["sen_flat"])
            sen_result.append(f"{sen_avg:.3}")
            flat_result.append(f"{flat_avg:.3}")
            maxprob_result.append(f"{maxprob_avg:.3}")
            sen_flat_result.append(f"{sen_flat_avg:.3}")
        print("SEN\t\t" + "\t".join(sen_result))
        print("FLAT\t\t" + "\t".join(flat_result))
        print("MAXPROB\t\t" + "\t".join(maxprob_result))
        print("SENFLAT\t\t" + "\t".join(sen_flat_result))


if __name__ == "__main__":
    path = "/home/wtan12/flatness/output"
    dataset = "dbpedia"
    for _, _, files in os.walk(path):
        for filetype in ["_", "-medium", "-large", "-xl"]:
            for file in files:
                if (
                    file.startswith(f"{dataset}_gpt2{filetype}")
                    and "abstain" in file
                    and "512" in file
                ):
                    # if file.startswith("sst2_gpt2-large"):
                    print(f"Checking result of {file}")
                    with open(f"{path}/{file}", "rb") as f:
                        data = pickle.load(f)
                        main(data, dataset)
                    break
        break
