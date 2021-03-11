"""
Input is a string, we split it into sentences
"""
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
