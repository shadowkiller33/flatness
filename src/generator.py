from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config
import torch
import numpy as np


class Generator:
    """wrapper class for generating logits"""

    def __init__(self, name) -> None:
        # TODO : make this general as a model card selection
        self.config = GPT2Config.from_pretrained(name, output_attentions=True)
        self.config.add_cross_attention
        self.tokenizer = GPT2Tokenizer.from_pretrained(name, pad_token="<|pad|>")
        self.model = GPT2LMHeadModel.from_pretrained(name, config=self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval().cuda()

    def get_logits(self, instruction):
        # TDOO: remove the hard-coded hyperparam with arguments
        token = self.tokenizer.encode(
            instruction,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=30,
        ).to("cuda")
        logits = self.model(token, return_dict=True)
        attention = logits["attentions"][0][-1].squeeze(0)
        return attention

    def validate(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            total_sample = 0
            correct = 0
            for batch in data_loader:
                sentence, label = batch
                sentence["input_ids"] = sentence["input_ids"].squeeze(1).cuda()
                sentence["attention_mask"] = sentence["attention_mask"].cuda()
                ss = label["input_ids"].squeeze(1)
                logits = self.model(**sentence).logits[:, -1, :]
                preds = torch.argmax(logits, dim=1)
                labels = ss[:, 0]
                correct += (np.array(preds.cpu()) == np.array(labels)).sum()
                total_sample += preds.shape[0]
            acc_total = correct / total_sample
        return acc_total

    def get_tokenizer(self):
        return self.tokenizer
