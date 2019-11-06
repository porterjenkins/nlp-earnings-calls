import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import random
from nltk.corpus import stopwords
import string

class DocGraph(object):
    stop = set(stopwords.words('english') + ['would', 'thing', 'question', 'could', ''])

    def __init__(self):
       self.doc_cntr = 0
       self.docs = {}
       self.edge_list = set()


    def add_node(self, doc):
        """

        :param doc: str
        :return:
        """
        if doc not in self.docs:
            self.docs[doc] = self.doc_cntr
            self.doc_cntr += 1


    def write(self):
        print("writing edge list and document codes")
        with open(cfg.vals["clean_data_dir"] + "doc-list.txt", 'w') as f:
            for text, idx in self.docs.items():
                text_clean = DocGraph.clean_doc(text)
                f.write("{}\t".format(idx))
                for char in text_clean:
                    f.write("{} ".format(char))
                f.write("\n")


        with open(cfg.vals["clean_data_dir"] + "edge-list.txt", 'w') as f:
            for edge in self.edge_list:
                f.write("{}, {}\n".format(edge[0], edge[1]))


    def add_edge(self, doc_1, doc_2):
        doc_1_id = self.docs[doc_1]
        doc_2_id = self.docs[doc_2]

        self.edge_list.add((doc_1_id, doc_2_id))


    def load_graph(self, fname, nrows=None):
        cntr = 0
        with open(fname, 'r') as f:
            for line in f:
                node_1, node_2 = line.split(", ")
                self.edge_list.add((int(node_1), int(node_2)))

                cntr += 1
                if nrows is not None:
                    if cntr == nrows:
                        break

    def load_docs(self, fname, nrows=None):
        cntr = 0
        with open(fname, 'r') as f:
            for line in f:
                idx, doc = line.split("\t")
                idx = int(idx)
                self.docs[idx] = doc

                cntr += 1
                if nrows is not None:
                    if cntr == nrows:
                        break

    def sample_edges(self, sample_size):
        edges = list(self.edge_list)
        samples = random.choices(edges, k=sample_size)

        return samples


    def gen_train_data(self, edges):

        print("Generating training data")
        with open(cfg.vals["clean_data_dir"] + "training.txt", 'w') as f:
            for edge in edges:
                try:
                    doc_1 = self.docs[edge[0]]
                    doc_2 = self.docs[edge[1]]
                except KeyError:
                    continue

                #doc_1_tokenized = DocGraph.clean_doc(doc_1)
                #doc_2_tokenized = DocGraph.clean_doc(doc_2)

                #for char in doc_1_tokenized:
                #    f.write(char + " ")
                #f.write("\t")

                f.write("{}\t{}".format(doc_1, doc_2))


    @classmethod
    def build_graph(cls, doc_graph, presentations, q_a):

        for speaker, vals in presentations.items():
            for speaker_2, vals_2 in q_a.items():
                for presentation_doc in vals['text']:
                    for q_a_doc in vals_2['text']:
                        doc_graph.add_edge(presentation_doc, q_a_doc)


    @classmethod
    def clean_doc(cls, doc):
        doc_tokenized = []
        for word in doc.split():
            word = word.translate(str.maketrans('', '', string.punctuation))
            word = word.lower()
            if word not in DocGraph.stop:
                doc_tokenized.append(word)
        return doc_tokenized





