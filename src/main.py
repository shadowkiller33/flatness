import argparse
from src.generator import Generator
from src.scorer import Scorer
from src.data_helper import DataHelper

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def main(args):
    np.random.seed(args.seed)
    # torch.manual_seed(seed)
    generator = Generator(args.model)
    data_helper = DataHelper(args.dataset, args.demo_shots, args.seed)
    scorer = Scorer(args.mode, args.batch_size, generator.get_tokenizer())
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
        prompt_with_demo, prompt_label = data_helper.get_in_context_prompt(prompt)

        acc = scorer.score_acc(generator, prompt_with_demo, prompt_label)
        print(f"Accuracy: {acc}")
        performance_all.append(acc)
    scorer.correlation(flatness_all, performance_all)


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
