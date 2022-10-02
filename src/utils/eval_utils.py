# this file is deprecated and not used anymore
import torch.utils.data as Data
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
