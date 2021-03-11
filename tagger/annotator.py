from typing import Dict, List

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)

from helper.annotator_utils import *
from helper.utils import TokenClassifier, align_predictions
import nltk
from helper.heuristics import heuristics

import logging

logging.disable(logging.WARNING)


class Annotator:
    def __init__(self, label_file, model_file_path, device):
        # get labels
        labels = []
        with open(label_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                labels.append(line.strip())
        self.labels = labels

        label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
        self.label_map = label_map
        num_labels = len(label_map)

        # get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_file_path)

        # get model config
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_file_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)}
        )

        # get model
        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_file_path,
            from_tf=False,
            config=config
        )

        if device == 'gpu':
            self.model.cuda()

        # init the token classifier
        self.token_classifier = TokenClassifier(
            tokenizer=self.tokenizer,
            model=self.model,
            labels=self.labels
        )

    def prepare_inputs(self, text: str) -> List[List[str]]:
        sents = get_sents(text)
        new_sents = [nltk.word_tokenize(sent) for sent in sents]
        return new_sents

    def annotate(self, text):
        inputs = self.prepare_inputs(text)
        preds, label_ids = self.token_classifier.classify_token(inputs)
        preds_list, _ = align_predictions(preds, label_ids, self.label_map)
        assert len(inputs) == len(preds_list)
        output = []
        for words, labels, label_id in zip(inputs, preds_list, label_ids):
            assert len(words) == len(labels) or len(label_id) == 512
            if len(words) != len(labels):
                max_len = len(words)
                while len(labels) < max_len:
                    labels.append('O')
            assert len(words) == len(labels)
            for word, label in zip(words, labels):
                output.append((word, label))
        output = heuristics(output)
        return output
