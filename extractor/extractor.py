# %%
from extractor_utils import *


class Extractor:
    def __init__(self, keywords_file, parameters_file):
        self.keywords = read_keywords(keywords_file)
        self.parameters = read_parameters(parameters_file)

    def extract(self, text):
        np.random.seed(666)
        filtered_sents, cleaned_filtered_sents = keywords_filtering(text, self.keywords)
        if len(filtered_sents) <= 30:
            out_p = np.array([1] * len(filtered_sents))
        else:
            group = len(filtered_sents) // 10
            init_p, init_n = self.parameters[group]
            out_p = CEmethod(cleaned_filtered_sents, N=init_n, init_p=init_p)
        samples = [np.random.binomial(1, p=out_p) for j in range(1)]
        extracted = get_text(samples[0], filtered_sents)
        return extracted
