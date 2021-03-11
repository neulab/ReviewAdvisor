# %% This is for automatically convert jsonl to sequence labeling
# how many sentences together is specified using parameter passing

import fire
import jsonlines
from transformers import AutoTokenizer
from typing import List
import nltk


def endswith(sent: str, extensions: List[str]):
    for extension in extensions:
        if sent.endswith(extension):
            return True
    return False


def contain_open_bracket(text: str):
    has_open_bracket = False
    for c in text:
        if c == '(':
            has_open_bracket = True
        if has_open_bracket and c == ')':
            has_open_bracket = False
    return has_open_bracket


def get_sents(text: str) -> List[str]:
    """ Give a text string, return the sentence list """
    # Here are some heuristics that we use to get appropriate sentence splitter.
    # 1. combine sentences with its successor when certain conditions satisfied
    sent_list: List[str] = nltk.tokenize.sent_tokenize(text)
    new_sent_list = [sent.replace("\n", "") for sent in sent_list]
    postprocessed = []
    buff = ""
    for sent in new_sent_list:
        if endswith(sent, ['i.e.', 'i.e .', 'e.g.', 'e.g .', 'resp.', 'resp .',
                           'et al.', 'et al .', 'i.i.d.', 'i.i.d .', 'Eq.',
                           'Eq .', 'eq.', 'eq .', 'incl.', 'incl .', 'Fig.',
                           'Fig .', 'w.r.t.', 'w.r.t .', 'sec.', 'sec .',
                           'Sec.', 'Sec .']) or len(sent) < 10 \
                or contain_open_bracket(sent):
            buff += sent
        else:
            postprocessed.append(buff + sent)
            buff = ""
    if len(buff) > 0:
        postprocessed.append(buff)
    return postprocessed


def get_aligned_data(json_line, tokenizer: AutoTokenizer):
    """ Get sentence tokens with its corresponding aspect tag.
        Return sth like this:
        [['ICLR_2017_1',
        ('this', 'clarity_positive'),
        ('paper','clarity_positive'),
        ('is', 'clarity_positive'),
        ('well', 'clarity_positive'),
        ('written', 'clarity_positive'),
        ('and', 'clarity_positive'),
        ('easy', 'clarity_positive'),
        ('to', 'clarity_positive'),
        ('follow', 'clarity_positive')],
        ...
        ]
    """
    paper_id: str = json_line.get('id')
    text: str = json_line.get('text')
    labels: List = json_line.get('labels')

    sents = get_sents(text)
    split_sent_list = []
    for sent in sents:
        split_sent_list.append(nltk.word_tokenize(sent))

    pointer = 0
    aligned_review = []
    for sent in split_sent_list:
        align_list = [paper_id]

        for token in sent:
            # We substitute the token if it cannot be tokenized by Bert
            current_subwords_len = len(tokenizer(token).get('input_ids'))
            if current_subwords_len == 0:
                token = 'sp_tok'

            start = pointer + text[pointer:].find(token)  # start of a token
            pointer = start + len(token)  # end of a token
            has_aspect = False
            for label_list in labels:
                label_start = label_list[0]
                label_end = label_list[1]
                label_text = label_list[2]

                if label_start <= start and pointer <= label_end:
                    align_list.append((token, label_text))
                    has_aspect = True
                    break

            if not has_aspect:
                align_list.append((token, 'O'))

        if len(align_list) > 1:
            aligned_review.append(align_list)

    return aligned_review


def concate_sentences(aligned_review: List, num: int):
    new_aligned_review = []
    paper_id = aligned_review[0][0]
    for i in range(0, len(aligned_review), num):
        align_list = [paper_id]
        sents = aligned_review[i: i + num]
        for sent in sents:
            align_list += sent[1:]
        new_aligned_review.append(align_list)
    return new_aligned_review


def split(json_line, tokenizer: AutoTokenizer, num: int):
    aligned_review = get_aligned_data(json_line, tokenizer)
    new_aligned_review = concate_sentences(aligned_review, num)
    return new_aligned_review


def read_jsonlines(jsonl_file):
    out = []
    with jsonlines.open(jsonl_file, 'r') as reader:
        for obj in reader:
            out.append(obj)
    return out


def reformat(sent_file):
    # relace 2 blank lines with 1 blank line
    with open(sent_file, "r", encoding="utf8") as f:
        data = f.read()
    data = data.replace("\n\n\n", "\n\n")
    with open(sent_file, "w", encoding="utf8") as f:
        f.write(data)


def main(jsonl_file, num: int, seqlab_file_name, id_file_name):
    """
    Given a jsonl file, format into seqlab file with id file.
    :param filename: jsonl file
    :param num: the number of sentences to concate together
    :param seqlab_file_name: the output seqlab format file
    :param id_file_name:  the output id file
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    out = []
    json_lines = read_jsonlines(jsonl_file)
    for json_line in json_lines:
        out += split(json_line, tokenizer, num)

    # write file
    # token and label is separated by white space
    id_list = []
    sent_list = []
    for data in out:
        # data: [id, (token, aspect), (token, aspect), ...]
        paper_id = data[0]
        id_list.append(paper_id)
        for elem in data[1:]:
            line = elem[0] + " " + elem[1]
            sent_list.append(line)
        sent_list.append('\n')

    id_file = open(id_file_name, 'w')
    for elem in id_list:
        print(elem, file=id_file)
    id_file.flush()

    seqlab_file = open(seqlab_file_name, 'w')
    for elem in sent_list:
        print(elem, file=seqlab_file)
    seqlab_file.flush()

    reformat(seqlab_file_name)


if __name__ == '__main__':
    # python split.py human_labeled.jsonl 1 review_with_aspect.txt id.txt
    fire.Fire(main)
