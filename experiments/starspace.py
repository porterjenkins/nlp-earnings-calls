import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from utils import load_word_embedding_tsv
from preprocessing.graph import DocGraph
import numpy as np


def get_doc_embedding(docs, embeddings, dim_size, write=True, fname=''):
    doc_embedding = {}
    cntr = 0
    for idx, doc in docs.items():
        word_list = doc.to_string().split()
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
    doc_graph.load_docs(cfg.vals["clean_data_dir"] + "doc-list.txt")

    words = load_word_embedding_tsv(cfg.vals["output_dir"] + 'starspace_words.tsv')
    doc_embedding = get_doc_embedding(doc_graph.docs_id, words, dim_size=cfg.vals['hidden_size'], write=True, fname=cfg.vals["output_dir"] + 'starspace_docs.txt')

