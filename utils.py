import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from operator import itemgetter
import random
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

null_words = ["N/A", "", "[MASK]"]
num_top_tokens = 100
num_gen_tokens = 1
all_regular_preds = []
all_calibrated_preds = []
all_answers = []


def train(train_loader, model, optimizer, EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch_num in range(EPOCHS):
        count = 0
        total_loss_train = 0
        for batch in train_loader:
            sentence, label = batch
            sentence["input_ids"] = sentence["input_ids"].squeeze(1).cuda()
            sentence["attention_mask"] = sentence["attention_mask"].cuda()
            ss = label["input_ids"].squeeze(1).cuda()
            aa = ss[:, 0]
            logits = model(**sentence).logits[:, -1, :]
            count += 1
            loss = criterion(logits, aa.long())
            optimizer.zero_grad()
            loss.backward()
            total_loss_train += loss
            optimizer.step()

        if epoch_num % 10 == 0:
            print(total_loss_train / count)


def validate(valloader, model):
    model.eval()
    with torch.no_grad():
        total_sample = 0
        correct = 0
        for batch in valloader:
            sentence, label = batch

            sentence["input_ids"] = sentence["input_ids"].squeeze(1).cuda()
            sentence["attention_mask"] = sentence["attention_mask"].cuda()
            ss = label["input_ids"].squeeze(1)
            logits = model(**sentence).logits[:, -1, :]
            preds = torch.argmax(logits, dim=1)
            labels = ss[:, 0]
            correct += (np.array(preds.cpu()) == np.array(labels)).sum()
            total_sample += preds.shape[0]
        acc_total = correct / total_sample
    return acc_total


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(
        self,
        dataset_text,
        dataset_label,
        tokenizer,
        max_seq_len,
        instruction,
        model,
        prompt,
    ):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        self.model = model
        self.ins = instruction
        self.trans_dist = {}
        self.prompt = prompt

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     #self.model.resize_token_embeddings(len(self.tokenizer))
        # text1 = self.ins + f'<|startoftext|> {self.prompt}Input: {self.text[idx]}<|pad|>Output:' + '\n'
        encode_result = self.tokenizer(
            self.text[idx],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=1000,
        )
        label = self.tokenizer(
            self.labels[idx], return_tensors="pt", padding="max_length", max_length=10
        )
        return encode_result, label


def lowercase_list(lst):
    return [l.lower() for l in lst]


def create_set(data, demo_num, seed, dict):
    from random import sample

    random.seed(seed)
    test_inputs = [x["text"] for x in data["test"]]
    test_labels = [dict[str(x["label"])] for x in data["test"]]
    index = random.sample(range(1, len(data["train"]["text"])), demo_num)
    # demos = sample(data['train']['text'],demo_num)
    # a = [-2, 1, 5, 3, 8, 5, 6]
    # b = [1, 2, 5]
    # print(itemgetter(*b)(a))
    text, label = data["train"]["text"], data["train"]["label"]
    demon_inputs = [text[i] for i in index]
    demon_labels = [label[i] for i in index]  # data['train']['label'][index]

    return demon_labels, demon_inputs, test_inputs, test_labels


def pashuffle(string, perc=10):
    data = string.split()
    L = len(data)
    gap = random.randint(1, 3)
    a = random.randint(0, L - 4)

    data[a], data[a + gap] = data[a + gap], data[a]
    return " ".join(data)
