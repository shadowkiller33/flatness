import torch.utils.data as Data
import torch
from functools import reduce
from src.utils.dataset_utils import loader_labeled
import numpy as np


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


def score(
    candidate, model, tokenizer, test_inputs, test_labels, eval_batch_size, prompt
):
    test_dataset = loader_labeled(
        test_inputs[:20], test_labels[:20], tokenizer, 1000, candidate, model, prompt
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset, batch_size=eval_batch_size, shuffle=False
    )
    acc_total = validate(test_loader, model)
    print(acc_total)
    return acc_total


def flat(input, mode="max"):
    if mode == "mean":
        avg_tensor = torch.mean(torch.stack(input))
        normalization = reduce(lambda x, y: x * y, list(avg_tensor.size()))
        mean = torch.sum(avg_tensor) / normalization
        return mean.detach().cpu().item()

    elif mode == "max":
        max_tensor = torch.max(torch.stack(input))
        return max_tensor.detach().cpu().item()

    elif mode == "avg_max":
        avg_tensor = torch.mean(torch.stack(input))
        max_tensor = torch.max(avg_tensor)
        return max_tensor.detach().cpu().item()
