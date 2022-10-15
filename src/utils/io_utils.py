import os
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
    f_p_list, f_s_list, f_k_list = [], [], []
    ours_MI_p_list, ours_MI_s_list, ours_MI_k_list = [], [], []
    ours_sen_p_list, ours_sen_s_list, ours_sen_k_list = [], [], []
    MI_sen_p_list, MI_sen_s_list, MI_sen_k_list = [], [], []

    for seed_id in table.keys():
        seed_level_info = table[seed_id]
        sen_p_list.append(seed_level_info["sen_p"])
        sen_s_list.append(seed_level_info["sen_s"])
        sen_k_list.append(seed_level_info["sen_k"])

        mi_p_list.append(seed_level_info["mi_p"])
        mi_s_list.append(seed_level_info["mi_s"])
        mi_k_list.append(seed_level_info["mi_k"])



        ours_MI_p_list.append(seed_level_info["ours_MI_p"])
        ours_MI_s_list.append(seed_level_info["ours_MI_s"])
        ours_MI_k_list.append(seed_level_info["ours_MI_k"])

        ours_sen_p_list.append(seed_level_info["ours_sen_p"])
        ours_sen_s_list.append(seed_level_info["ours_sen_s"])
        ours_sen_k_list.append(seed_level_info["ours_sen_k"])

        MI_sen_p_list.append(seed_level_info["MI_sen_p"])
        MI_sen_s_list.append(seed_level_info["MI_sen_s"])
        MI_sen_k_list.append(seed_level_info["MI_sen_k"])


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
    ours_MI_p_avg, ours_MI_p_var = np.average(np.array(ours_MI_p_list)), np.var(
        np.array(ours_MI_p_list)
    )
    ours_MI_s_avg, ours_MI_s_var = np.average(np.array(ours_MI_s_list)), np.var(
        np.array(ours_MI_s_list)
    )
    ours_MI_k_avg, ours_MI_k_var = np.average(np.array(ours_MI_k_list)), np.var(
        np.array(ours_MI_k_list)
    )


    MI_sen_p_avg, MI_sen_p_var = np.average(np.array(MI_sen_p_list)), np.var(
        np.array(MI_sen_p_list)
    )
    MI_sen_s_avg, MI_sen_s_var = np.average(np.array(MI_sen_s_list)), np.var(
        np.array(MI_sen_s_list)
    )
    MI_sen_k_avg, MI_sen_k_var = np.average(np.array(MI_sen_k_list)), np.var(
        np.array(MI_sen_k_list)
    )


    ours_sen_p_avg, ours_sen_p_var = np.average(np.array(ours_sen_p_list)), np.var(
        np.array(ours_sen_p_list)
    )
    ours_sen_s_avg, ours_sen_s_var = np.average(np.array(ours_sen_s_list)), np.var(
        np.array(ours_sen_s_list)
    )
    ours_sen_k_avg, ours_sen_k_var = np.average(np.array(ours_sen_k_list)), np.var(
        np.array(ours_sen_k_list)
    )

    print("Avg/Var of sensitivity's correlation to performance:")
    print(f"Pearson Correlation: {sen_p_avg:.4}/{sen_p_var:.4}")
    print(f"Spearman Correlation: {sen_s_avg:.4}/{sen_s_var:.4}")
    print(f"Kendalltau Correlation: {sen_k_avg:.4}/{sen_k_var:.4}")
    print()
    print("Avg/Var of mutual information's correlation to performance:")
    print(f"Pearson Correlation: {mi_p_avg:.4}/{mi_p_var:.4}")
    print(f"Spearman Correlation: {mi_s_avg:.4}/{mi_s_var:.4}")
    print(f"Kendalltau Correlation: {mi_k_avg:.4}/{mi_k_var:.4}")
    print()
    print("Avg/Var of (flatness + MI) correlation to performance:")
    print(f"Pearson Correlation: {ours_MI_p_avg:.4}/{ours_MI_p_var:.4}")
    print(f"Spearman Correlation: {ours_MI_s_avg:.4}/{ours_MI_s_var:.4}")
    print(f"Kendalltau Correlation: {ours_MI_k_avg:.4}/{ours_MI_k_var:.4}")
    print()

    print("Avg/Var of (flatness + sen) correlation to performance:")
    print(f"Pearson Correlation: {ours_sen_p_avg:.4}/{ours_sen_p_var:.4}")
    print(f"Spearman Correlation: {ours_sen_s_avg:.4}/{ours_sen_s_var:.4}")
    print(f"Kendalltau Correlation: {ours_sen_k_avg:.4}/{ours_sen_k_var:.4}")
    print()

    print("Avg/Var of (sen + MI) correlation to performance:")
    print(f"Pearson Correlation: {MI_sen_p_avg:.4}/{MI_sen_p_var:.4}")
    print(f"Spearman Correlation: {MI_sen_s_avg:.4}/{MI_sen_s_var:.4}")
    print(f"Kendalltau Correlation: {MI_sen_k_avg:.4}/{MI_sen_k_var:.4}")
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
        print(
            f"Origin: {avg_acc:.4}/{var_acc:.4}\tCalibrated: {avg_acc_c:.4}/{var_acc_c:.4}"
        )
