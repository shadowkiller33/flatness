from utils import *
import torch.utils.data as Data
from functools import reduce

def score(candidate,model,tokenizer,test_inputs, test_labels,eval_batch_size,prompt):
    test_dataset = loader_labeled(
        test_inputs[:20], test_labels[:20], tokenizer, 1000, candidate, model,prompt)
    test_loader = Data.DataLoader(
        dataset=test_dataset, batch_size=eval_batch_size, shuffle=False)
    acc_total = validate(test_loader, model)
    print(acc_total)
    return acc_total

def flat(input, mode = 'max'):
    if mode == 'mean':
        avg_tensor = torch.mean(torch.stack(input))
        normalization = reduce(lambda x, y: x*y, list(avg_tensor.size()))
        mean = torch.sum(avg_tensor)/normalization
        return mean.detach().cpu().item()

    elif mode == 'max':
        max_tensor = torch.max(torch.stack(input))
        return max_tensor.detach().cpu().item()

    elif mode == 'avg_max':
        avg_tensor = torch.mean(torch.stack(input))
        max_tensor = torch.max(avg_tensor)
        return max_tensor.detach().cpu().item()




