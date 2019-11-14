
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_doc_embedding, load_word_embedding_tsv
import config.config as cfg
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from utils import load_docs

K = 4

starspace = load_doc_embedding(cfg.vals["output_dir"] + 'starspace_docs.txt')
starspace_mtx = pd.DataFrame.from_dict(starspace, orient='index').values
docs = load_docs(cfg.vals["clean_data_dir"] + "raw-doc-list.txt")

idx_map = dict(zip(range(starspace_mtx.shape[0]), docs.keys()))

query_docs = [4016, 4018, 4037, 4074, 5864, 5867, 11926, 11930, 11944, 11988, 37319, 58, 199]
search = NearestNeighbors(n_neighbors=K)
search.fit(starspace_mtx)


for q in query_docs:
    try:
        vec = starspace[q].reshape(1,-1)
        #X = np.concatenate([vec, starspace], axis=0)
        #print(X.shape)
        dist, idx = search.kneighbors(vec)
        with open("doc-to-doc-{}.txt".format(q), 'w') as f:
            f.write("query: {}\n".format(docs[q]))
            for cntr, i in enumerate(idx[0][1:]):
                doc = docs[idx_map[i]]
                f.write("neigbor {}:\t{}\n".format(cntr, doc))



    except KeyError:
        print(q + " not in dictionary")