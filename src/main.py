import argparse
from ast import Return
from src.utils.eval_utils import sensitivity_compute
from src.generator import Generator
from src.scorer import Scorer
from src.data_helper import DataHelper
from src.utils.io_utils import print_results, load_pickle, save_pickle
from copy import deepcopy
import logging
import numpy as np
import torch
from scipy.special import entr
from scipy.special import softmax
import random
import sys

logger = logging.getLogger(__name__)


def main(args):
    default_params = {
        "conditioned_on_correct_classes": True,
        "subsample_test_set": args["subsample_test_set"],
        "api_num_log_prob": args["api_num_log_prob"],
        "approx": args["approx"],
        "bs": args["bs"],
        "mode": args["mode"],
        "data_dir": args["data_dir"],
        "perturbed_num": args["perturbed_num"],
    }
    model = args["models"]
    # list of all experiment parameters to run
    all_params = []
    for dataset in args["datasets"]:
        for num_shots in args["all_shots"]:
            out_dir = f"{args['output_dir']}/{dataset}"
            data_helper = DataHelper(args["data_dir"], dataset)
            for prompt_id, prompt in enumerate(data_helper.get_prompts(dataset)):
                for seed in args["seeds"]:
                    p = deepcopy(default_params)
                    p["prompt_id"] = prompt_id
                    p["prompt"] = prompt
                    p["seed"] = seed
                    p["dataset"] = dataset
                    p["num_shots"] = num_shots
                    p[
                        "expr_name"
                    ] = f"{p['dataset']}_{model}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_prompt{prompt_id}_subsample_seed{seed}"
                    p["output_dir"] = out_dir
                    all_params.append(p)

    if args["use_saved_results"]:
        load_results(all_params, model)
    else:
        save_results(all_params, model)


def load_results(params_list, model_name):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        data_helper = DataHelper(
            params["data_dir"], params["dataset"], params["num_shots"], params["seed"]
        )
        for prompt_id, prompt in enumerate(data_helper.get_prompts()):
            saved_result = load_pickle(prompt_id, params)
            keys = [params["dataset"], params["model"], params["num_shots"]]
            node = result_tree  # root
            for k in keys:
                if not (k in node.keys()):
                    node[k] = dict()
                node = node[k]
            node[params["seed"]] = saved_result["accuracies"]
            print_results(result_tree)


def compute_flatness():
    # difference = []
    # attention = generator.get_logits(prompt)
    #
    # for perturbed_sentence in perturbed:
    #     attention_p = generator.get_logits(perturbed_sentence)
    #     difference.append(torch.abs(attention_p - attention))
    #
    # flatness = scorer.flat(difference)
    # flatness_all.append(flatness)
    pass


def update_result_dict(table, prompt_id, seed, prompt, entry_name, result):
    if seed not in table:
        prompt_info = {"id": prompt_id, "prompt": prompt, entry_name: result}
        table[seed] = {prompt_id: prompt_info}
    else:
        if prompt_id not in table[seed]:
            table[seed][prompt_id] = {
                "id": prompt_id,
                "prompt": prompt,
                entry_name: result,
            }
        else:
            table[seed][prompt_id][entry_name] = result


def save_results(params_list, model_name):
    result_tree = dict()
    result_table = {}  # keep all sens, flatness, mi results
    generator = Generator(model_name)
    for params in params_list:
        print(f"Evaluate on promt id: {params['prompt_id']}, seed: {params['seed']}")
        data_helper = DataHelper(params["data_dir"], params["dataset"])
        scorer = Scorer(params["mode"], params["bs"], generator.get_tokenizer())

        # the current prompt we evaluate metrics on
        prompt_id, prompt = params["prompt_id"], params["prompt"]
        seed = params["seed"]

        # append demos for predictions
        (
            train_sentences,
            train_labels,
            test_sentences,
            test_labels,
        ) = data_helper.get_in_context_prompt(params, prompt, seed)
        raw_resp_test = generator.get_model_response(
            params, train_sentences, train_labels, test_sentences
        )
        all_label_probs = generator.get_label_probs(
            params, raw_resp_test, train_sentences, train_labels, test_sentences
        )
        original_labels = np.argmax(all_label_probs, axis=1)
        normalized_probs = softmax(all_label_probs, axis=1)
        avg_prob = np.average(normalized_probs, axis=0)
        entropy1 = np.average(entr(normalized_probs).sum(axis=1))
        entropy2 = entr(avg_prob).sum()
        mutual_info = entropy2 - entropy1
        # update result table
        update_result_dict(result_table, prompt_id, seed, prompt, "mi", mutual_info)

        content_free_inputs = ["N/A", "", "[MASK]"]
        output = []

        perturbed = DataHelper.get_pertubed_set(
            prompt, params["perturbed_num"]
        )  # get perturbed data
        prompt_orders = DataHelper.get_prompt_order(
            train_sentences, train_labels, params["perturbed_num"]
        )
        for (perturbed_prompt, order) in zip(perturbed, prompt_orders):
            (
                _,
                _,
                test_sentences,
                test_labels,
            ) = data_helper.get_in_context_prompt(params, perturbed_prompt)
            train_sentences, train_labels = order
            raw_resp_test = generator.get_model_response(
                params, train_sentences, train_labels, test_sentences
            )
            all_label_probs111 = generator.get_label_probs(
                params, raw_resp_test, train_sentences, train_labels, test_sentences
            )
            labels111 = np.argmax(all_label_probs111, axis=1)
            # sensitivity = np.sum([labels111 == original_labels])/len(train_labels)
            output.append(labels111)
        sensitivity = sensitivity_compute(output, original_labels)
        update_result_dict(result_table, prompt_id, seed, prompt, "sen", sensitivity)

        p_cf = generator.get_p_content_free(
            params,
            train_sentences,
            train_labels,
            content_free_inputs=content_free_inputs,
        )
        acc_original = 0  #%scorer.eval_accuracy(all_label_probs, test_labels)
        acc_calibrated = scorer.eval_accuracy(
            all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf
        )
        accuracies = [acc_original, acc_calibrated]
        print(f"Accuracies: {accuracies}")
        print(f"p_cf      : {p_cf}")
        update_result_dict(
            result_table, prompt_id, seed, prompt, "perf", acc_calibrated
        )

        # TODO: need better ways to handle the print tree
        # # serialize results for reproduction
        # keys = [params["dataset"], params["model"], params["num_shots"]]
        # node = result_tree  # root
        # for k in keys:
        #     if not (k in node.keys()):
        #         node[k] = dict()
        #     node = node[k]
        # node[params["seed"]] = accuracies

        # # save to file
        # result_to_save = dict()
        # params_to_save = deepcopy(params)
        # result_to_save["prompt"] = prompt
        # result_to_save["params"] = params_to_save
        # result_to_save["train_sentences"] = train_sentences
        # result_to_save["train_labels"] = train_labels
        # result_to_save["test_sentences"] = test_sentences
        # result_to_save["test_labels"] = test_labels
        # result_to_save["raw_resp_test"] = raw_resp_test
        # result_to_save["all_label_probs"] = all_label_probs
        # result_to_save["p_cf"] = p_cf
        # result_to_save["accuracies"] = accuracies
        # if "prompt_func" in result_to_save["params"].keys():
        #     params_to_save["prompt_func"] = None
        # save_pickle(prompt_id, params, result_to_save)

    # Evaluate Result using saved dictionaries
    # scorer.Flatness_correlation(flatness_all, performance_all)
    print(result_table)
    sen_corr, mi_corr, ours_corr = [], [], []
    for seed_id in result_table.keys():
        sen_list, mi_list, perf_list = [], [], []
        for prompt_id in result_table[seed_id]:
            sen_list.append(result_table[seed_id][prompt_id]["sen"])
            mi_list.append(result_table[seed_id][prompt_id]["mi"])
            perf_list.append(result_table[seed_id][prompt_id]["perf"])
        sen_corr.append(scorer.sen_correlation(sen_list, perf_list))
        mi_corr.append(scorer.MI_correlation(mi_list, perf_list))
        ours_corr.append(scorer.ours_correlation(sen_list, mi_list, perf_list))
    # calculate avg and variance of correlations
    pearson_list = []
    spearman_list = []
    kendalltau_list = []
    for item in sen_corr:
        pearson_list.append(item[0])
        spearman_list.append(item[1])
        kendalltau_list.append(item[2])
    print(
        f"Avg/Var pearson correlation for Sensitivity is {np.average(np.array(pearson_list))}/{np.var(np.array(pearson_list))}"
    )
    print(
        f"Avg/Var pearson correlation for Spearman is {np.average(np.array(spearman_list))}/{np.var(np.array(spearman_list))}"
    )
    print(
        f"Avg/Var pearson correlation for Kendalltau is {np.average(np.array(kendalltau_list))}/{np.var(np.array(kendalltau_list))}"
    )
    # print_results(result_tree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument(
        "--models",
        dest="models",
        action="store",
        required=True,
        help="name of model(s), e.g., GPT2-XL",
    )
    parser.add_argument(
        "--datasets",
        dest="datasets",
        action="store",
        required=True,
        help="name of dataset(s), e.g., agnews",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_seeds",
        dest="num_seeds",
        action="store",
        required=True,
        help="num seeds for the training set",
        type=int,
    )
    parser.add_argument(
        "--perturbed_num",
        dest="perturbed_num",
        action="store",
        required=True,
        help="num of samples for the perturbed set",
        type=int,
    )
    parser.add_argument(
        "--all_shots",
        dest="all_shots",
        action="store",
        required=True,
        help="num training examples to use",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        action="store",
        required=True,
        default="max",
        help="the way to calculate flatness",
    )
    # other arguments
    parser.add_argument(
        "--subsample_test_set",
        dest="subsample_test_set",
        action="store",
        required=False,
        type=int,
        default=None,
        help="size of test set to use to speed up eval. None means using all test set",
    )
    parser.add_argument(
        "--api_num_log_prob",
        dest="api_num_log_prob",
        action="store",
        required=False,
        type=int,
        default=100,
        help="number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API",
    )
    parser.add_argument(
        "--bs",
        dest="bs",
        action="store",
        required=False,
        type=int,
        default=None,
        help="batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.",
    )
    # flags
    parser.add_argument(
        "--use_saved_results",
        dest="use_saved_results",
        action="store_const",
        const=True,
        default=False,
        help="whether to load the results from pickle files and not run the model",
    )
    parser.add_argument(
        "--approx",
        dest="approx",
        action="store_const",
        const=True,
        default=False,
        help="whether to set token prob to zero if not in top 100",
    )
    parser.add_argument("--data-dir", required=True, type=str)

    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args["models"] = convert_to_list(args["models"])[0]
    args["datasets"] = convert_to_list(args["datasets"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)
    seeds = []
    while len(set(seeds)) < int(args["num_seeds"]):
        seeds.append(random.randint(1, 100))
    args["seeds"] = seeds
    main(args)
