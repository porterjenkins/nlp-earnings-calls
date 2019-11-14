import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_doc_embedding, load_word_embedding_tsv
import config.config as cfg
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from utils import load_docs

K = 5

starspace = load_doc_embedding(cfg.vals["output_dir"] + 'starspace_docs.txt')
starspace = pd.DataFrame.from_dict(starspace, orient='index').values
docs = load_docs(cfg.vals["clean_data_dir"] + "raw-doc-list.txt")

words = load_word_embedding_tsv(cfg.vals["output_dir"] + 'starspace_words.tsv')

query_words = ['volatility', 'revenue', 'income', 'cost', 'earnings', 'marketing', 'loss', 'bank', 'technology', 'ipod', \
               'internet', 'supply', 'demand', 'profitability', 'seasonality', 'demand high', 'demand low', 'cost increase', 'cost decrease', 'revenue decrease', 'revenue increase']
search = NearestNeighbors(n_neighbors=K)
search.fit(starspace)

idx_map = dict(zip(range(starspace.shape[0]), docs.keys()))


for w in query_words:
    w = w.split()
    if len(w) > 1:
        vec = np.zeros((1, cfg.vals['hidden_size']))
        for i in w:

            vec += words[i].reshape(1, -1)

    else:
        vec = words[w[0]].reshape(1, -1)
    #X = np.concatenate([vec, starspace], axis=0)
    #print(X.shape)
    dist, idx = search.kneighbors(vec)
    print("---- {} ----".format(w))
    for i in idx[0]:
        doc = docs[idx_map[i]]
        print(doc)


