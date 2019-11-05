import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from utils import load_docs
from preprocessing.conference_call_analysis_parallel import clean
import numpy as np


docs = load_docs(cfg.vals["clean_data_dir"] + "doc-list.txt")

docs_tokenized = []

cntr = 0
n_docs = len(docs)

for idx, doc in docs.items():
    token = clean(doc).split()
    docs_tokenized.append(token)
    cntr += 1

    progress = cntr / n_docs
    print("Preprocessing text: {:.4f} done".format(progress), end="\r")



dictionary_pre = Dictionary(docs_tokenized)
corpus = [dictionary_pre.doc2bow(doc) for doc in docs_tokenized]
model = LdaModel(corpus, num_topics=cfg.vals['hidden_size'])


with open(cfg.vals["output_dir"] + "lda.txt", 'w') as f:
    cntr = 0
    for doc in corpus:
        topics = model[doc]
        embed = np.zeros(cfg.vals['hidden_size'])
        f.write("{}\t".format(cntr))
        for topic_prob in topics:
            embed[topic_prob[0]] = topic_prob[1]

        for component in embed:
            f.write("{} ".format(component))
        f.write("\n")

        cntr += 1

