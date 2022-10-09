from curses.ascii import isalpha, isdigit
import os
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np


def load_pickle(path, name):
    # load saved results from model
    file_name = os.path.join(path, name)
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data


def save_pickle(path, name, data):
    # save results from model
    if not os.path.exists(path):
        Path(path).mkdir(parents=True)
    file_name = os.path.join(path, name)
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, "wb") as file:
        pickle.dump(data, file)
    print(f"Result Table Saved to {file_name}")
    return data


def print_results(table):
    # print only top level stats (avg/var correlation and accuracy)
    # calculate correlations across seeds
    sen_p_list, sen_s_list, sen_k_list = [], [], []
    mi_p_list, mi_s_list, mi_k_list = [], [], []
    ours_p_list, ours_s_list, ours_k_list = [], [], []

    for seed_id in table.keys():
        seed_level_info = table[seed_id]
        sen_p_list.append(seed_level_info["sen_p"])
        sen_s_list.append(seed_level_info["sen_s"])
        sen_k_list.append(seed_level_info["sen_k"])
        mi_p_list.append(seed_level_info["mi_p"])
        mi_s_list.append(seed_level_info["mi_s"])
        mi_k_list.append(seed_level_info["mi_k"])
        ours_p_list.append(seed_level_info["ours_p"])
        ours_s_list.append(seed_level_info["ours_s"])
        ours_k_list.append(seed_level_info["ours_k"])
    sen_p_avg, sen_p_var = np.average(np.array(sen_p_list)), np.var(
        np.array(sen_p_list)
    )
    sen_s_avg, sen_s_var = np.average(np.array(sen_s_list)), np.var(
        np.array(sen_s_list)
    )
    sen_k_avg, sen_k_var = np.average(np.array(sen_k_list)), np.var(
        np.array(sen_k_list)
    )
    mi_p_avg, mi_p_var = np.average(np.array(mi_p_list)), np.var(np.array(mi_p_list))
    mi_s_avg, mi_s_var = np.average(np.array(mi_s_list)), np.var(np.array(mi_s_list))
    mi_k_avg, mi_k_var = np.average(np.array(mi_k_list)), np.var(np.array(mi_k_list))
    ours_p_avg, ours_p_var = np.average(np.array(ours_p_list)), np.var(
        np.array(ours_p_list)
    )
    ours_s_avg, ours_s_var = np.average(np.array(ours_s_list)), np.var(
        np.array(ours_s_list)
    )
    ours_k_avg, ours_k_var = np.average(np.array(ours_k_list)), np.var(
        np.array(ours_k_list)
    )
    print("Avg/Var of sensitivity's correlation to performance:")
    print(f"Pearson Correlation: {sen_p_avg}/{sen_p_var}")
    print(f"Spearman Correlation: {sen_s_avg}/{sen_s_var}")
    print(f"Kendalltau Correlation: {sen_k_avg}/{sen_k_var}")
    print()
    print("Avg/Var of mutual information's correlation to performance:")
    print(f"Pearson Correlation: {mi_p_avg}/{mi_p_var}")
    print(f"Spearman Correlation: {mi_s_avg}/{mi_s_var}")
    print(f"Kendalltau Correlation: {mi_k_avg}/{mi_k_var}")
    print()
    print("Avg/Var of our's correlation to performance:")
    print(f"Pearson Correlation: {ours_p_avg}/{ours_p_var}")
    print(f"Spearman Correlation: {ours_s_avg}/{ours_s_var}")
    print(f"Kendalltau Correlation: {ours_k_avg}/{ours_k_var}")
    print()

    # calculate accuracy across seeds
    acc_result = {}
    for seed_id in table.keys():
        for prompt_id in table[seed_id].keys():
            if isinstance(prompt_id, int):
                prompt_level_info = table[seed_id][prompt_id]
                origin_acc = prompt_level_info["acc"]
                calibrated_acc = prompt_level_info["acc_c"]
                if prompt_id not in acc_result:
                    acc_result[prompt_id] = {
                        "prompt": prompt_level_info["prompt"],
                        "acc": [origin_acc],
                        "acc_c": [calibrated_acc],
                    }
                else:
                    acc_result[prompt_id]["acc"].append(origin_acc)
                    acc_result[prompt_id]["acc_c"].append(calibrated_acc)

    for p_id in acc_result.keys():
        prompt_data = acc_result[p_id]
        prompt = prompt_data["prompt"]
        origin_acc = prompt_data["acc"]
        avg_acc, var_acc = np.average(np.array(origin_acc)), np.var(
            np.array(origin_acc)
        )
        calibrated_acc = prompt_data["acc_c"]
        avg_acc_c, var_acc_c = np.average(np.array(calibrated_acc)), np.var(
            np.array(calibrated_acc)
        )

        print(f"Average/Variance of accuracy for Prompt: {prompt}")
        print(f"Origin: {avg_acc}/{var_acc}\tCalibrated: {avg_acc_c}/{var_acc_c}")
