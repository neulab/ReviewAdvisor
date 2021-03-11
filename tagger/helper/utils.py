# coding=utf-8
# Adapted from Huggingface token-classification task

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedTokenizer, AutoModelForTokenClassification


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
):
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()

        # Truncate long sequence
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        # pad on the right
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            {'input_ids': torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
             'attention_mask': torch.tensor(input_mask, dtype=torch.long).unsqueeze(0),
             'token_type_ids': torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0),
             'labels': torch.tensor(label_ids, dtype=torch.long).unsqueeze(0)}
        )
    return features


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


class TokenClassifier:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model: AutoModelForTokenClassification,
            labels: List[str],
            max_seq_length=512
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

    def prepare_features(self, sents: List[List[str]]):
        input_examples = []
        for i, sent in enumerate(sents):
            labels = ['O'] * len(sent)
            input_examples.append(InputExample(guid=f'{i}', words=sent, labels=labels))

        features = convert_examples_to_features(
            examples=input_examples,
            label_list=self.labels,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id
        )
        return features

    def classify_token(self, sents: List[List[str]]):
        features = self.prepare_features(sents)

        preds: torch.Tensor = None
        label_ids: torch.Tensor = None

        for inputs in features:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                step_eval_loss, logits = outputs[:2]

            # output predictions
            if preds is None:
                preds = logits.detach()
            else:
                preds = torch.cat((preds, logits.detach()), dim=0)

            if inputs.get('labels') is not None:
                if label_ids is None:
                    label_ids = inputs['labels'].detach()
                else:
                    label_ids = torch.cat((label_ids, inputs['labels'].detach()))

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
        return preds, label_ids
