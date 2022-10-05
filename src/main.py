import argparse
from ast import Return
from src.generator import Generator
from src.scorer import Scorer
from src.data_helper import DataHelper
from src.utils.io_utils import print_results, load_pickle, save_pickle
from copy import deepcopy
import logging
import numpy as np
import torch

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
        "seed": args["num_seeds"],
    }

    # list of all experiment parameters to run
    all_params = []
    for model in args["models"]:
        for dataset in args["datasets"]:
            for num_shots in args["all_shots"]:
                p = deepcopy(default_params)
                p["model"] = model
                p["dataset"] = dataset
                p["num_shots"] = num_shots
                p[
                    "expr_name"
                ] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                all_params.append(p)

    if args["use_saved_results"]:
        load_results(all_params)
    else:
        save_results(all_params)


def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params["dataset"], params["model"], params["num_shots"]]
        node = result_tree  # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params["seed"]] = saved_result["accuracies"]
    print_results(result_tree)


def save_results(params_list):
    result_tree = dict()
    for params in params_list:
        generator = Generator(params["model"])
        data_helper = DataHelper(
            params["data_dir"], params["dataset"], params["num_shots"], params["seed"]
        )
        scorer = Scorer(params["mode"], params["bs"], generator.get_tokenizer())

        performance_all = []
        flatness_all = []
        for prompt in DataHelper.get_prompts():
            perturbed = DataHelper.get_pertubed_set(prompt)
            difference = []
            attention = generator.get_logits(prompt)

            for perturbed_sentence in perturbed:
                attention_p = generator.get_logits(perturbed_sentence)
                difference.append(torch.abs(attention_p - attention))

            flatness = scorer.flat(difference)
            flatness_all.append(flatness)

            # append demos for predictions
            (
                train_sentences,
                train_labels,
                test_sentences,
                test_labels,
            ) = data_helper.get_in_context_prompt(params, prompt)
            raw_resp_test = generator.get_model_response(
                params, train_sentences, train_labels, test_sentences
            )
            all_label_probs = generator.get_label_probs(
                params, raw_resp_test, train_sentences, train_labels, test_sentences
            )
            # calculate P_cf
            content_free_inputs = ["N/A", "", "[MASK]"]
            p_cf = generator.get_p_content_free(
                params,
                train_sentences,
                train_labels,
                content_free_inputs=content_free_inputs,
            )
            acc_original = scorer.eval_accuracy(all_label_probs, test_labels)
            acc_calibrated = scorer.eval_accuracy(
                all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf
            )
            accuracies = [acc_original, acc_calibrated]
            print(f"Accuracies: {accuracies}")
            print(f"p_cf      : {p_cf}")
            performance_all.append(acc_calibrated)

            keys = [params["dataset"], params["model"], params["num_shots"]]
            node = result_tree  # root
            for k in keys:
                if not (k in node.keys()):
                    node[k] = dict()
                node = node[k]
            node[params["seed"]] = accuracies

            # save to file
            result_to_save = dict()
            params_to_save = deepcopy(params)
            result_to_save["params"] = params_to_save
            result_to_save["train_sentences"] = train_sentences
            result_to_save["train_labels"] = train_labels
            result_to_save["test_sentences"] = test_sentences
            result_to_save["test_labels"] = test_labels
            result_to_save["raw_resp_test"] = raw_resp_test
            result_to_save["all_label_probs"] = all_label_probs
            result_to_save["p_cf"] = p_cf
            result_to_save["accuracies"] = accuracies
            if "prompt_func" in result_to_save["params"].keys():
                params_to_save["prompt_func"] = None
            save_pickle(params, result_to_save)
        scorer.correlation(flatness_all, performance_all)
        print_results(result_tree)


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
        "--num_seeds",
        dest="num_seeds",
        action="store",
        required=True,
        help="num seeds for the training set",
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
        "--method",
        dest="method",
        action="store_const",
        type=str,
        default='flatness',
        help="two options: flatness or mutual information",
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

    args["models"] = convert_to_list(args["models"])
    args["datasets"] = convert_to_list(args["datasets"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)
    main(args)
