from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config
import torch
import numpy as np
from src.utils.eval_utils import construct_prompt, chunk_size_helper, chunks
from tqdm import tqdm


class Generator:
    """wrapper class for generating logits"""

    def __init__(self, name) -> None:
        # TODO : make this general as a model card selection
        print(f"Loading model {name}")
        self.model = GPT2LMHeadModel.from_pretrained(name, output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.eval().cuda()

    def get_logits(self, instruction):
        # TDOO: remove the hard-coded hyperparam with arguments
        token = self.tokenizer.encode(
            instruction,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=50,
        ).to("cuda")

        logits = self.model(token, return_dict=True)
        attention = [
            logits["attentions"][i].squeeze(0) for i in range(len(logits["attentions"]))
        ]  # logits["attentions"][-1]
        cat = torch.cat(attention)
        return cat

    def get_p_content_free(
        self, params, train_sentences, train_labels, content_free_inputs=("N/A",)
    ):
        """Query model with content free input, return its prediction probability for each label"""
        label_dict = params["label_dict"]

        all_p_y = []
        for content_free_input in content_free_inputs:
            prompt = construct_prompt(
                params, train_sentences, train_labels, content_free_input
            )

            p_y = [0] * len(label_dict)
            for i, answers in label_dict.items():
                prob = 0
                for a in answers:
                    prob += np.exp(
                        self.complete_gpt2(
                            prompt + " " + a,
                            l=0,
                            echo=True,
                            num_log_probs=1,
                        )["choices"][0]["logprobs"]["token_logprobs"][-1]
                    )
                p_y[i] = prob
            all_p_y.append(p_y)

        p_y = np.mean(np.array(all_p_y), axis=0)
        p_y = p_y / np.sum(p_y)  # normalize
        return p_y

    def get_label_probs(
        self, params, raw_resp, train_sentences, train_labels, test_sentences
    ):
        """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
        num_classes = len(params["label_dict"])
        approx = params["approx"]
        assert len(raw_resp) == len(test_sentences)

        # Fill in the labels that is in the top k prob
        all_label_probs = []
        all_missing_positions = []
        for i, ans in tqdm(enumerate(raw_resp), disable=True):
            top_logprobs = ans["logprobs"]["top_logprobs"][
                0
            ]  # [0] since we only ask for complete one more token
            label_probs = [0] * len(params["label_dict"].keys())
            for j, label_list in params["label_dict"].items():
                all_found = True
                for (
                    label
                ) in label_list:  # each possible label correspond to the same class
                    label = " " + label  # notice prompt does not have space after 'A:'
                    if label in top_logprobs:
                        label_probs[j] += np.exp(top_logprobs[label])
                    else:
                        all_found = False
                if not all_found:
                    position = (i, j)  # (which test example, which label)
                    all_missing_positions.append(position)
            all_label_probs.append(label_probs)
        all_label_probs = np.array(all_label_probs)  # prob not normalized

        # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
        # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
        if (not approx) and (len(all_missing_positions) > 0):
            print(
                f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}"
            )
            all_additional_prompts = []
            num_prompts_each = []
            for position in all_missing_positions:
                which_sentence, which_label = position
                test_sentence = test_sentences[which_sentence]
                label_list = params["label_dict"][which_label]
                for label in label_list:
                    prompt = construct_prompt(
                        params, train_sentences, train_labels, test_sentence
                    )
                    prompt += " " + label
                    all_additional_prompts.append(prompt)
                num_prompts_each.append(len(label_list))

            # chunk the prompts and feed into model
            chunked_prompts = list(
                chunks(all_additional_prompts, chunk_size_helper(params))
            )
            all_probs = []
            for chunk_id, chunk in enumerate(chunked_prompts):
                resp = self.complete_gpt2(chunk, l=0, echo=True, num_log_probs=1)
                for ans in resp["choices"]:
                    prob = np.exp(ans["logprobs"]["token_logprobs"][-1])
                    all_probs.append(prob)

            assert sum(num_prompts_each) == len(all_probs)
            assert len(num_prompts_each) == len(all_missing_positions)

            # fill in corresponding entries in all_label_probs
            for index, num in enumerate(num_prompts_each):
                probs = []
                while num > 0:
                    probs.append(all_probs.pop(0))
                    num -= 1
                prob = np.sum(probs)
                i, j = all_missing_positions[index]
                all_label_probs[i][j] = prob

            assert len(all_probs) == 0, "all should be popped"
            assert (
                all_label_probs > 0
            ).all(), "all should be populated with non-zero value"

        return all_label_probs  # NOT NORMALIZED

    def get_model_response(
        self,
        params,
        train_sentences,
        train_labels,
        test_sentences,
        perturbed=False,
        return_all_prompts=False,
        num_tokens_to_predict_override=None,
        override_prompt=None,
    ):
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
                prompts.append(
                    construct_prompt(
                        params, train_sentences, train_labels, test_sentence
                    )
                )
        else:
            prompts = override_prompt

        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
        for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
            if num_tokens_to_predict_override is not None:
                num_tokens_to_predict = num_tokens_to_predict_override
            else:
                num_tokens_to_predict = params["num_tokens_to_predict"]
            resp = self.complete_gpt2(
                test_chunk_prompts,
                l=num_tokens_to_predict,
                perturbed=perturbed,
                num_log_probs=params["api_num_log_prob"],
            )

            for answer_id, answer in enumerate(resp["choices"]):
                all_raw_answers.append(answer)
        if return_all_prompts:
            return all_raw_answers, prompts
        else:
            return all_raw_answers

    def complete_gpt2(
        self, prompt, l=10, perturbed=False, num_log_probs=None, echo=False
    ):
        """This function runs GPT-2 locally but places the outputs into an json that looks just like the one
        provided by the OpenAI API."""
        if isinstance(prompt, str):
            prompt = [prompt]  # the code below assumes a list
        input_ids = self.tokenizer.batch_encode_plus(
            prompt, return_tensors="pt", padding=True
        )

        # Add perturbation
        if perturbed == True:
            with torch.no_grad():
                for param in self.model.parameters():
                    p = torch.randn(param.size()) * 0.0001
                    param.add_(p.cuda())

        # greedily generate l tokens
        if l > 0:
            # the generate function can handle left padded inputs automatically in HF
            # total_sequences is now the input + possible generated output
            total_sequences = self.model.generate(
                input_ids=input_ids["input_ids"].cuda(),
                attention_mask=input_ids["attention_mask"].cuda(),
                max_length=l + len(input_ids["input_ids"][0]),
                do_sample=False,
            )
        else:
            assert echo == True and l == 0
            total_sequences = input_ids["input_ids"].cuda()

        # they want the probs of the top tokens
        if num_log_probs is not None:
            # we are left padding, so we need to adjust the position IDs
            attention_mask = (total_sequences != 50256).float()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # get the logits for the context and the next l tokens
            logits = (
                self.model.forward(
                    input_ids=total_sequences,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
                .logits.detach()
                .cpu()
            )
            if not echo:
                # get the top tokens and probs for the generated l tokens
                probs = torch.softmax(logits[:, -l - 1 :], dim=2).cpu()
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
                curr_json["text"] = self.tokenizer.decode(
                    total_sequences[batch_id][-l:], skip_special_tokens=True
                )
            else:
                curr_json["text"] = self.tokenizer.decode(
                    total_sequences[batch_id], skip_special_tokens=True
                )

            # fill the return json with the top tokens and probs to match the OpenAI return value.
            if num_log_probs is not None:
                curr_json["logprobs"] = {}
                curr_json["logprobs"]["top_logprobs"] = []
                curr_json["logprobs"]["token_logprobs"] = []
                curr_json["logprobs"]["tokens"] = []
                if not echo:
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for (
                        current_element_top_log_probs,
                        current_element_top_tokens,
                    ) in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                        # tokens is a list of the top token at each position
                        curr_json["logprobs"]["tokens"].append(
                            self.tokenizer.decode([current_element_top_tokens[0]])
                        )
                        # token_logprobs is a list of the logprob of the top token at each position
                        curr_json["logprobs"]["token_logprobs"].append(
                            current_element_top_log_probs[0].item()
                        )
                        # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                        temp = {}
                        for log_prob, token in zip(
                            current_element_top_log_probs, current_element_top_tokens
                        ):
                            temp[self.tokenizer.decode(token.item())] = log_prob.item()
                        curr_json["logprobs"]["top_logprobs"].append(temp)
                else:
                    # same as not above but small tweaks
                    # we add null to the front because for the GPT models, they have null probability for the first token
                    # (for some reason they don't have an beginning of sentence token)
                    curr_json["logprobs"]["top_logprobs"].append("null")
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for index, (
                        current_element_top_log_probs,
                        current_element_top_tokens,
                    ) in enumerate(
                        zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])
                    ):
                        # skip padding tokens
                        if total_sequences[batch_id][index].item() == 50256:
                            continue
                        temp = {}
                        for log_prob, token in zip(
                            current_element_top_log_probs, current_element_top_tokens
                        ):
                            temp[self.tokenizer.decode(token.item())] = log_prob.item()
                        curr_json["logprobs"]["top_logprobs"].append(temp)
                    for index in range(len(probs[batch_id])):
                        curr_json["logprobs"]["tokens"].append(
                            self.tokenizer.decode([total_sequences[batch_id][index]])
                        )
                    curr_json["logprobs"]["token_logprobs"].append("null")
                    for index, log_probs_token_position_j in enumerate(
                        logprobs[batch_id][:-1]
                    ):
                        # probs are left shifted for LMs
                        curr_json["logprobs"]["token_logprobs"].append(
                            log_probs_token_position_j[
                                total_sequences[batch_id][index + 1]
                            ]
                        )

            choices.append(curr_json)
        return_json["choices"] = choices
        return return_json

    def get_tokenizer(self):
        return self.tokenizer
