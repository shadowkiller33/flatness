from curses.ascii import isdigit
import os
import pickle
from src.scorer import Scorer
from src.utils.io_utils import print_results


def compute_corr(result_table):
    scorer = Scorer("mean", 1, None)
    verbose = False

    for seed_id in result_table.keys():
        sen_list, mi_list, flat_list, perf_list = [], [], [], []
        for prompt_id in result_table[seed_id].keys():
            if isinstance(prompt_id, int):
                sen_list.append(result_table[seed_id][prompt_id]["sen"])
                mi_list.append(result_table[seed_id][prompt_id]["mi"])
                flat_list.append(result_table[seed_id][prompt_id]["flat"])
                perf_list.append(result_table[seed_id][prompt_id]["perf"])
        # sensitivity
        sen_p, sen_s, sen_k = scorer.sen_correlation(
            sen_list, perf_list, verbose=verbose
        )
        result_table[seed_id]["sen_p"] = sen_p
        result_table[seed_id]["sen_s"] = sen_s
        result_table[seed_id]["sen_k"] = sen_k

        # MI
        mi_p, mi_s, mi_k = scorer.MI_correlation(mi_list, perf_list, verbose=verbose)
        result_table[seed_id]["mi_p"] = mi_p
        result_table[seed_id]["mi_s"] = mi_s
        result_table[seed_id]["mi_k"] = mi_k

        # Flat
        f_p, f_s, f_k = scorer.ours_correlation(flat_list, perf_list, verbose=verbose)
        result_table[seed_id]["f_p"] = f_p
        result_table[seed_id]["f_s"] = f_s
        result_table[seed_id]["f_k"] = f_k

        # Flat + MI
        ours_MI_p, ours_MI_s, ours_MI_k = scorer.ours_correlation_MI(
            mi_list, flat_list, perf_list, verbose=verbose
        )
        result_table[seed_id]["ours_MI_p"] = ours_MI_p
        result_table[seed_id]["ours_MI_s"] = ours_MI_s
        result_table[seed_id]["ours_MI_k"] = ours_MI_k

        # Flat + sensitivity
        ours_sen_p, ours_sen_s, ours_sen_k = scorer.ours_correlation_sen(
            sen_list, flat_list, perf_list, verbose=verbose
        )
        result_table[seed_id]["ours_sen_p"] = ours_sen_p
        result_table[seed_id]["ours_sen_s"] = ours_sen_s
        result_table[seed_id]["ours_sen_k"] = ours_sen_k

        # MI + sensitivity
        MI_sen_p, MI_sen_s, MI_sen_k = scorer.MI_sen_correlation(
            sen_list, mi_list, perf_list, verbose=verbose
        )
        result_table[seed_id]["MI_sen_p"] = MI_sen_p
        result_table[seed_id]["MI_sen_s"] = MI_sen_s
        result_table[seed_id]["MI_sen_k"] = MI_sen_k
    print_results(result_table)


if __name__ == "__main__":
    path = "/home/wtan12/flatness/output"
    for _, _, files in os.walk(path):
        for file in files:
            if file.startswith("dbpedia_gpt2_") and "abstain" in file:
                # if file.startswith("sst2_gpt2-large"):
                print(f"Checking result of {file}")
                with open(f"{path}/{file}", "rb") as f:
                    data = pickle.load(f)
                    print(data)
                    # compute_corr(data)
                break
        break
