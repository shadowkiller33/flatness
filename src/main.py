import argparse
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.dataset_utils import create_set, pashuffle
from src.utils.eval_utils import score, flat
from scipy.stats import spearmanr, pearsonr, kendalltau
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)
# from promptsource.templates import DatasetTemplates
# ag_news_prompts = DatasetTemplates('ag_news')
def main(args):
    batch_size = args.batch_size
    demo_num = args.demo_shots
    seed = args.seed
    data = args.dataset
    name = args.model
    mode = args.mode

    np.random.seed(seed)
    # torch.manual_seed(seed)

    datasets = load_dataset(data)
    dict = {"0": "world", "1": "sports", "2": "business", "3": "science"}
    demo_labels, demo_inputs, test_inputs, test_labels = create_set(
        datasets, demo_num, seed, dict
    )
    instruction1 = ""
    ag_news_prompts = [
        "What label best describes this news article?",
        "What is this a piece of news regarding for?",
        " What is the category of the following news?",
        "Which is the most relevant topic of the following news?",
        "Give the topic of the given text.",
        "Read the text below, provide its focused topic.",
        "Is this a piece of news regarding world, sport, business,or science?",
        "Which section of a newspaper would this article likely appear in?",
    ]

    config = GPT2Config.from_pretrained("gpt2", output_attentions=True)
    config.add_cross_attention
    tokenizer = GPT2Tokenizer.from_pretrained(name, pad_token="<|pad|>")
    # tokenizer1 = GPT2Tokenizer.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name, config=config)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # test_dataset = loader_labeled(
    # test_inputs, test_labels, tokenizer, 1000, instruction1,model)
    model.resize_token_embeddings(len(tokenizer))
    model.eval().cuda()

    performance_all = []
    flatness_all = []
    for prompt in ag_news_prompts:
        perturbed = set()
        instruction1 = prompt
        difference = []
        ids = tokenizer.encode(
            instruction1,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=30,
        ).to("cuda")
        logits = model(ids, return_dict=True)
        attention = logits["attentions"][0][-1].squeeze(0)
        while len(perturbed) < 10:
            sss = pashuffle(instruction1, perc=10)
            if sss + "\n" != instruction1:
                perturbed.add(sss)
        for perturbed_sentence in perturbed:
            ids1 = tokenizer.encode(
                perturbed_sentence,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=30,
            ).to("cuda")
            logits1 = model(ids1, return_dict=True)
            attention_p = logits1["attentions"][0][-1].squeeze(0)
            difference.append(torch.abs(attention_p - attention))
        flatness = flat(difference, mode)
        flatness_all.append(flatness)
        for x, y in zip(demo_labels, demo_inputs):
            instruction1 += "\n" + "Input:" + y + "\n" + "Output:" + dict[str(x)]
        for i in range(len(test_inputs)):
            test_inputs[i] = (
                instruction1 + "\n" + "Input:" + test_inputs[i] + "\n" + "Output:"
            )
        acc = score(
            instruction1, model, tokenizer, test_inputs, test_labels, batch_size, prompt
        )
        performance_all.append(acc)

    # correlation between flatness and model performance evaluation
    a = pearsonr(flatness_all, performance_all)[0]
    b = spearmanr(flatness_all, performance_all)[0]
    c = kendalltau(flatness_all, performance_all)[0]
    logger.info(
        "On {}, The pearson correlation between {} flatness and human score is {}".format(
            name, mode, a
        )
    )
    logger.info(
        "On {}, The spearman correlation between {} flatness and human score is {}".format(
            name, mode, b
        )
    )
    logger.info(
        "On {}, The kendall correlation between {} flatness and human score is {}".format(
            name, mode, c
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Take arguments from commandline")
    parser.add_argument("--mode", default="max", help="the mode to calculate flatness")
    parser.add_argument(
        "--demo-shots",
        default=16,
        type=int,
        help="Type number of demos in the prompt if applicable",
    )
    parser.add_argument("--model", default="gpt2", type=str, help="model name")
    parser.add_argument("--batch-size", default=4, type=int, help="batch-size")
    parser.add_argument("--dataset", default="ag_news", type=str, help="dataset name")
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Type in seed that changes sampling of examples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
