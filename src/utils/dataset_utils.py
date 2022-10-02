import random
from torch.utils.data import Dataset


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(
        self,
        dataset_text,
        dataset_label,
        tokenizer,
        max_seq_len,
    ):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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
