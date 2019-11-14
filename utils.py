import numpy as np

def load_docs(fname, nrows=None):
    docs = {}
    cntr = 0
    with open(fname, 'r') as f:
        for line in f:
            idx, doc = line.split("\t")
            idx = int(idx)
            docs[idx] = doc

            cntr += 1
            if nrows is not None:
                if cntr == nrows:
                    break

    return docs



def load_word_embedding_tsv(fname, nrows=None):

    embeddings = {}
    with open(fname, 'r') as f:
        for cntr, line in enumerate(f):
            split = line.split("\t")
            #word, vec = line.split(" ")
            word = split[0]
            vec = np.array(split[1:], dtype=np.float32)
            embeddings[word] = vec

            if nrows is not None and cntr == nrows:
                break

    return embeddings

def load_doc_embedding(fname, nrows=None):

    embeddings = {}
    with open(fname, 'r') as f:
        for cntr, line in enumerate(f):
            idx, vec = line.split("\t")
            #word, vec = line.split(" ")
            idx = int(idx)
            vec = vec.strip().split(" ")
            vec = np.array(vec, dtype=np.float32)
            embeddings[idx] = vec

            if nrows is not None and cntr == nrows:
                break

    return embeddings
