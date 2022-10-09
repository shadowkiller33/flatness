import os
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np


def load_pickle(prompt_id, params):
    # load saved results from model
    output_dir = f"{params['output_dir']}/{prompt_id}"
    file_name = os.path.join(output_dir, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data


def save_pickle(prompt_id, params, data):
    # save results from model
    output_dir = f"{params['output_dir']}/{prompt_id}"
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True)
    file_name = os.path.join(output_dir, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, "wb") as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data


def print_results(tree, names=("Original Accuracy  ", "Calibrated Accuracy")):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(
                    zip(
                        accuracies_mean, accuracies_low, accuracies_high, accuracies_std
                    )
                ):
                    print(
                        f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}"
                    )
                print()
