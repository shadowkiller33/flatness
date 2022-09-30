import json
from tqdm import tqdm
from transformers import pipeline
import transformers as hf
import torch.utils.data as Data
import argparse
from utils import *
from datasets import load_dataset
from performance import *
# Load prompts for this dataset

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--mode', default="max", help='the mode to calculate flatness')
parser.add_argument('--demo-shots', default=4, type=int, help='Type number of demos in the prompt if applicable')
parser.add_argument('--model', default='gpt2', type=str, help='model name')
parser.add_argument('--batch-size', default=4, type=int, help='batch-size')
parser.add_argument('--dataset', default='ag_news', type=str, help='dataset name')
parser.add_argument('--seed', type=int, help='Type in seed that changes sampling of examples')
args = parser.parse_args()

# from promptsource.templates import DatasetTemplates
# ag_news_prompts = DatasetTemplates('ag_news')
batch_size = args.batch_size
demo_num = args.demo_shots
seed = args.seed
data = args.dataset
name = args.model
mode = args.mode

np.random.seed(seed)
torch.manual_seed(seed)
datasets = load_dataset(data)
dict = {'0':'world','1':'sports','2':'business','3':'science'}
demo_labels, demo_inputs, test_inputs, test_labels = create_set(datasets,demo_num,seed,dict)
instruction1 = ''
ag_news_prompts = ['What label best describes this news article?\n','What is this a piece of news regarding for?\n',' What is the category of the following news?\n', 'What is this news about?\n','Give the topic of the given text.\n','Read the text below, provide its focused topic.\n','Is this a piece of news regarding world, sport, business,or science?\n', 'Which section of a newspaper would this article likely appear in?\n']
config = GPT2Config.from_pretrained("gpt2", output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained(name)
model = GPT2LMHeadModel.from_pretrained(name, config=config)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))
test_dataset = loader_labeled(
            test_inputs, test_labels, tokenizer, 1000, instruction1,model)

model.eval().cuda()
performance_all = []
flatness_all = []
for prompt in ag_news_prompts:
    perturbed = set()
    instruction1 = prompt
    difference = []
    ids = tokenizer.encode(instruction1, return_tensors='pt').to('cuda')
    attention = model(ids,return_dict=True)['attentions'][0][-1].squeeze(0)
    while len(perturbed) < 10:
        sss = pashuffle(instruction1, perc=10) + '\n'
        if sss != prompt:
            perturbed.add(sss)
    for perturbed_sentence in perturbed:
        ids = tokenizer.encode(perturbed_sentence, return_tensors='pt').to('cuda')
        attention_p = model(ids, return_dict=True)['attentions'][0][-1].squeeze(0)
        difference.append(torch.abs(attention_p-attention))
    flatness = flat(difference,mode)
    flatness_all.append(flatness)
    for x,y in zip(demo_labels, demo_inputs):
       instruction1 += '\n'+ "Input:" + y + '\n' + 'Output:' + dict[str(x)]

    acc = score(instruction1,model,tokenizer,test_inputs, test_labels,batch_size)
    performance_all.append(acc)


from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau
a = (pearsonr(flatness, performance_all)[0])
b = (spearmanr(flatness, performance_all)[0])
c = (kendalltau(flatness, performance_all)[0])
print('On {}, The pearson correlation between {} flatness and human score is {}'.format(mode, a))
print('On {}, The spearman correlation between {} flatness and human score is {}'.format(mode, b))
print('On {}, The kendall correlation between {} flatness and human score is {}'.format(mode, c))


