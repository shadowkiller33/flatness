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
from transformers import pipeline, set_seed
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
            sentence['input_ids'] = sentence['input_ids'].squeeze(1).cuda()
            sentence['attention_mask'] = sentence['attention_mask'].cuda()
            ss = label['input_ids'].squeeze(1).cuda()
            aa = ss[:,0]
            logits = model(**sentence).logits[:, -1, :]
            count += 1
            loss = criterion(logits,aa.long())
            optimizer.zero_grad()
            loss.backward()
            total_loss_train += loss
            optimizer.step()

        if epoch_num % 10 == 0:
            print(total_loss_train/count)


def validate(valloader, model):
    model.eval()
    with torch.no_grad():
        total_sample = 0
        correct = 0
        for batch in valloader:
            sentence, label = batch

            sentence['input_ids'] = sentence['input_ids'].cuda()
            sentence['attention_mask'] = sentence['attention_mask'].cuda()
            ss = label['input_ids'].squeeze(1)
            output = model(**sentence, return_dict=True)
            logit = output.logits[0][-1:]
            preds = torch.argmax(logit,dim=1)
            labels = ss[:, 0]
            correct += (np.array(preds.cpu()) ==  np.array(labels)).sum()
            total_sample += preds.shape[0]
        acc_total = correct/total_sample
    return acc_total



class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, instruction,model,prompt):
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
        #text1 = self.ins + f'<|startoftext|> {self.prompt}Input: {self.text[idx]}<|pad|>Output:' + '\n'
        encode_result = self.tokenizer(self.text[idx], return_tensors='pt')
        label = self.tokenizer(self.labels[idx], return_tensors='pt')
        return encode_result, label



def lowercase_list(lst):
    return [l.lower() for l in lst]


def create_set(data,demo_num,seed,dict):
    from random import sample
    random.seed(seed)
    test_inputs = [x['text'] for x in data['test']]
    test_labels = [dict[str(x['label'])] for x in data['test']]
    index = random.sample(range(1, len(data['train']['text'])), demo_num)
    #demos = sample(data['train']['text'],demo_num)
    # a = [-2, 1, 5, 3, 8, 5, 6]
    # b = [1, 2, 5]
    # print(itemgetter(*b)(a))
    text, label = data['train']['text'], data['train']['label']
    demon_inputs = [text[i] for i in index]
    demon_labels = [label[i] for i in index] #data['train']['label'][index]

    return demon_labels, demon_inputs, test_inputs, test_labels

def pashuffle(string, perc=10):
    data = string.split()
    L = len(data)
    gap = 1
    a = random.randint(0, L-2)

    data[a], data[a+gap] = data[a+gap], data[a]
    return " ".join(data)

def match(test_inputs, test_labels,model,tokenizer1):
    count = 0
    acc = 0
    pred = []
    labels = []
    for (sentence,label) in zip(test_inputs,test_labels):
        generation = model.predict_next(sentence, 1)
        pred.append(generation[0])
        labels.append(label)
        if generation[0].lower() == label:
            acc += 1
        count += 1
    print(acc/count)
    return acc/count


import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
# import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta',
                                       'davinci-beta']
            return 20
    else:
        return bs


def random_sampling(sentences, labels, num):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)


gpt2_model = None
gpt2_tokenizer = None


def setup_gpt2(model_name):
    # load the GPT-2 model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-2 model")
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        gpt2_model.eval().cuda()

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")


# def setup_gpt3():
#     # get OpenAI access key
#     with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
#         key = f.readline().strip()
#         openai.api_key = key

def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)

    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(),
                                              attention_mask=input_ids['attention_mask'].cuda(),
                                              max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()

    # they want the probs of the top tokens
    if num_log_probs is not None:
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (total_sequences != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # get the logits for the context and the next l tokens
        logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids,
                                    return_dict=True).logits.detach().cpu()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:, -l - 1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
        else:
            curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1],
                                                                                     top_tokens[batch_id][:-1]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append('null')
                # cutoff the -1 here because the probs are shifted one over for LMs
                for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(
                        zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                    # skip padding tokens
                    if total_sequences[batch_id][index].item() == 50256:
                        continue
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(len(probs[batch_id])):
                    curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append('null')
                for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                    # probs are left shifted for LMs
                    curr_json['logprobs']['token_logprobs'].append(
                        log_probs_token_position_j[total_sequences[batch_id][index + 1]])

        choices.append(curr_json)
    return_json['choices'] = choices
    return return_json


# def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
#     # call GPT-3 API until result is provided and then return it
#     response = None
#     received = False
#     while not received:
#         try:
#             response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
#                                                 logprobs=num_log_probs, echo=echo, stop='\n', n=n)
#             received = True
#         except:
#             error = sys.exc_info()[0]
#             if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
#                 print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
#                 assert False
#
#             print("API error:", error)
#             time.sleep(1)
#     return response

def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'gpt2' in model:
        assert n == None  # unsupported at the moment
        assert temp == 0  # unsupported at the moment
        setup_gpt2(model)
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo)
    # else:
    #     setup_gpt3()
    #     return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)


def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function.
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l,
                                                                       np.int64):  # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str)  # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1]  # GPT models do not want a trailing space, so we cut off -1
    return prompt


def get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
    else:
        prompts = override_prompt

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        if num_tokens_to_predict_override is not None:
            num_tokens_to_predict = num_tokens_to_predict_override
        else:
            num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'],
                        num_log_probs=params['api_num_log_prob'])
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers


def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data


def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data


def print_results(tree, names=('Original Accuracy  ', 'Calibrated Accuracy')):
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
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()


def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree  # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)