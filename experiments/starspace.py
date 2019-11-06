import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from utils import load_docs
from preprocessing.graph import DocGraph
import numpy as np

def load_word_embedding(fname, nrows=None):

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


def get_doc_embedding(docs, embeddings, dim_size, write=True, fname=''):
    doc_embedding = {}
    cntr = 0
    for idx, doc in docs.items():
        word_list = doc.split()
        doc_matrix = np.zeros((len(word_list), dim_size))
        for i, word in enumerate(word_list):
            try:
                doc_matrix[i, :] = embeddings[word]
            except:
                pass

        doc_embedding[idx] = doc_matrix.sum(axis=0)

        if write:
            if cntr == 0:
                write_flag = 'w'
            else:
                write_flag = 'a'

            with open(fname, write_flag) as f:
                f.write("{}\t".format(idx))
                for i in doc_embedding[idx]:
                    f.write("{} ".format(i))
                f.write("\n")


        cntr += 1




    return doc_embedding




if __name__ == "__main__":

    doc_graph = DocGraph()
    doc_graph.load_docs(cfg.vals["clean_data_dir"] + "doc-list.txt", 1000)

    words = load_word_embedding(cfg.vals["output_dir"] + 'starspace_words.tsv', 5000)
    doc_embedding = get_doc_embedding(doc_graph.docs, words, dim_size=100, write=True, fname=cfg.vals["output_dir"] + 'starspace_docs.txt')
    print(doc_embedding)

