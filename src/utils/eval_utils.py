# this file is deprecated and not used anymore
import numpy as np


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def sensitivity_compute(output, original_label):
    L = len(output[0])
    num = len(output)
    output = np.array(output)
    #original_labels = np.array(original_label)
    all = 0
    for i in range(L):
        slice = output[:,i].tolist()
        target = original_label[i]
        ss = [x == target for x in slice]
        match = sum(ss)
        all += match
    sensitivity = all/L
    return sensitivity

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params["bs"]
    if bs is None:
        if "gpt2" in params["model"]:
            return 1
        else:
            assert params["model"] in [
                "ada",
                "babbage",
                "curie",
                "davinci",
                "ada-beta",
                "babbage-beta",
                "curie-beta",
                "davinci-beta",
            ]
            return 20
    else:
        return bs


def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function.
    if ("prompt_func" in params.keys()) and (params["prompt_func"] is not None):
        return params["prompt_func"](
            params, train_sentences, train_labels, test_sentence
        )

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if (
            isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64)
        ):  # integer labels for classification
            assert params["task_format"] == "classification"
            l_str = (
                params["label_dict"][l][0]
                if isinstance(params["label_dict"][l], list)
                else params["label_dict"][l]
            )
        else:
            assert isinstance(l, str)  # string labels
            assert params["task_format"] == "qa"
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == " "
    prompt += a_prefix[:-1]  # GPT models do not want a trailing space, so we cut off -1
    return prompt
