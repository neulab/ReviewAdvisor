# %%
import json
import sys
from collections import Counter
from typing import List

import nltk
import numpy as np
import traceback
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

sys.setrecursionlimit(1000000)
nltk.download('stopwords')
nltk.download('punkt')
stemming = PorterStemmer()
stops = set(stopwords.words("english"))


# first read keywords table
def read_keywords(keywords_file):
    keywords = []
    with open(keywords_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            keywords += line.split(" ")
    return keywords


# then read parameters table
def read_parameters(parameters_file):
    parameters = []
    with open(parameters_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            _, init_p, init_n = line.split(' ')
            parameters.append((float(init_p), int(init_n)))
    return parameters


def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X


def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks down into words
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""

    # Convert to lower case
    text = raw_text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]
    # Remove stop words
    meaningful_words = [w for w in stemmed_words if not w in stops]
    # Rejoin meaningful stemmed words
    joined_words = (" ".join(meaningful_words))
    # Return cleaned data
    return joined_words


def get_full_text(paper_json):
    full_text = ""
    with open(paper_json, 'r', encoding='utf8') as f:
        content_dict = json.loads(f.read())
        sections = content_dict.get('metadata').get('sections')
        for section in sections:
            heading: str = section.get('heading')
            text: str = section.get('text')
            if heading is not None:
                if heading.upper().__contains__('ACKNOW') or heading.upper().__contains__('APPEN'):
                    break
            if text is not None and len(text) > 0:
                full_text += text + " "
    full_text = full_text.replace("\n", " ").encode("utf-8", "ignore").decode("utf-8").strip()
    return full_text


# look how the filtering works
def get_sents(text: str) -> (List, List):
    """ give a text string, return the sentence list """
    # Here are some heuristics that we use to get appropriate sentence splitter.
    # 1. Delete sentences that are fewer than 25 characters.
    # 2. If a sentence ends in et al. Then concate with the sentence behind it.
    sent_list: List[str] = nltk.tokenize.sent_tokenize(text)
    new_sent_list = [sent.replace("\n", "") for sent in sent_list]
    postprocessed = []
    buff = ""
    for sent in new_sent_list:
        if sent.endswith('et al.') or sent.endswith('Eq.') \
                or sent.endswith('i.e.') or sent.endswith('e.g.'):
            buff += sent
        else:
            if len(buff + sent) > 25 and \
                    not (buff + sent).__contains__('arxiv') and \
                    not (buff + sent).__contains__('http'):
                postprocessed.append(buff + sent)
            buff = ""
    if len(buff) > 0:
        postprocessed.append(buff)
    cleaned_sent_list = apply_cleaning_function_to_list(postprocessed[:250])
    return postprocessed[:250], cleaned_sent_list


def keywords_filtering(text: str, keywords: List[str]) -> (List[str], List[str]):
    sents, cleaned_sents = get_sents(text)
    filtered_sents = []
    cleaned_filtered_sents = []
    for sent, clean_sent in zip(sents, cleaned_sents):
        words = nltk.word_tokenize(sent)
        for word in words:
            if word in keywords:
                filtered_sents.append(sent)
                cleaned_filtered_sents.append(clean_sent)
                break
    return filtered_sents, cleaned_filtered_sents


def score(sample: np.array, sent_list: List[str]) -> float:
    final_text = get_text(sample, sent_list)
    return get_score(final_text)


def get_text(sample: np.array, sent_list: List[str]) -> str:
    final_text = ""
    for idx in range(0, len(sample)):
        if sample[idx] == 1:
            final_text += sent_list[idx] + " "
    final_text = final_text.strip()
    return final_text


def get_score(text: str) -> float:
    words = nltk.word_tokenize(text)
    summ_len = len(words)
    counter = Counter(words)
    v = np.array(list(counter.values())) / summ_len
    return float(np.matmul(-v, np.log2(v)))


def isAllZeroOrOne(array):
    """ Use to check convergence """
    for elem in array:
        if elem != 1.0 and elem != 0.0:
            return False
    return True


def CEmethod(sent_list: List[str], N=10000, init_p=0.5, rho=0.05, alpha=0.7, iter=100) -> np.array:
    try:
        p = np.array([init_p] * len(sent_list))
        early_stop_step = 0
        gamma_old = 0.0
        for i in range(iter):
            if i >= 1:
                N = 1000
            samples = [np.random.binomial(1, p=p) for j in range(N)]
            scored_samples = [(sample, score(sample, sent_list)) for sample in samples if sample.sum() <= 30]

            while len(scored_samples) == 0:
                samples = [np.random.binomial(1, p=p) for j in range(N)]
                scored_samples = [(sample, score(sample, sent_list)) for sample in samples if sample.sum() <= 30]

            # np.quantile does not require a sorted input
            gamma = np.quantile([x[1] for x in scored_samples], 1 - rho)

            valid_samples = [sample[0] for sample in scored_samples if sample[1] >= gamma]

            # Relax the gamma a little bit due to floating point precision issue
            closeness = 0.0000000000001
            while len(valid_samples) == 0:
                valid_samples = [sample[0] for sample in scored_samples if sample[1] >= gamma - closeness]
                closeness *= 10

            new_p = sum(valid_samples) / len(valid_samples)

            if gamma == gamma_old:
                early_stop_step += 1

            p = alpha * p + (1 - alpha) * new_p
            gamma_old = gamma

            if early_stop_step >= 3 or isAllZeroOrOne(p):
                break
        return p

    except:
        return np.array([0] * len(sent_list))
