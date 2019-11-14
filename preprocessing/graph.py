import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import random
from documents.document import Document


class DocGraph(object):

    def __init__(self):
       self.doc_cntr = 0
       self.docs = {}
       self.docs_id = {}
       self.edge_list = set()


    def get_doc(self, text):
        return self.docs.get(text, None)


    def add_node(self, doc, text, min_token_cnt=3):
        """

        :param doc: str
        :return:
        """
        if doc not in self.docs and len(doc.tokens) > min_token_cnt:
            doc.set_id(self.doc_cntr)
            self.docs[text] = doc
            self.docs_id[doc.id] = doc
            self.doc_cntr += 1


    def write(self):

        n_docs = len(self.docs)
        with open(cfg.vals["clean_data_dir"] + "doc-list.txt", 'w') as f:
            with open(cfg.vals["clean_data_dir"] + "doc-meta-data.csv", 'w') as f2:
                with open(cfg.vals["clean_data_dir"] + "raw-doc-list.txt", 'w') as f3:
                    cntr = 0
                    for text, doc in self.docs.items():
                        f.write("{}\t".format(doc.id))
                        for char in doc.tokens:
                            f.write("{} ".format(char))
                        f.write("\n")

                        f2.write("{}, {}, {}\n".format(doc.id, doc.speaker, doc.speaker_type))
                        print("writing document codes: {:.2f}%".format(cntr / n_docs), end='\r')

                        f3.write("{}\t{}\n".format(doc.id, doc.text))


                        cntr += 1
                    print("done")

        n_edges = len(self.edge_list)
        cntr = 0
        with open(cfg.vals["clean_data_dir"] + "edge-list.txt", 'w') as f:
            for edge in self.edge_list:
                f.write("{}, {}\n".format(edge[0].id, edge[1].id))
                print("writing edge list: {:.2f}%".format(cntr / n_edges), end='\r')
                cntr += 1
            print("done")


    def add_edge(self, doc_1, doc_2):
        if doc_1 in self.docs and doc_2 in self.docs:
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
                idx, text = line.split("\t")
                idx = int(idx)
                doc = Document('')
                doc.set_tokens(text)
                doc.set_id(idx)
                self.docs_id[idx] = doc

                cntr += 1
                if nrows is not None:
                    if cntr == nrows:
                        break

    def sample_edges(self, sample_size):
        edges = list(self.edge_list)
        samples = random.choices(edges, k=sample_size)

        return samples


    def gen_train_data(self, edges):

        cntr = 0
        n_edges = len(edges)
        with open(cfg.vals["clean_data_dir"] + "training.txt", 'w') as f:
            for edge in edges:

                doc_1 = edge[0]
                doc_2 = edge[1]


                f.write("{}\t{}\n".format(doc_1.to_string(), doc_2.to_string()))

                print("Generating training data: {:.2f}%".format(cntr / n_edges), end='\r')
                cntr += 1

            print("done")

    @classmethod
    def build_graph(cls, doc_graph, presentations, q_a):

        for speaker, vals in presentations.items():
            for speaker_2, vals_2 in q_a.items():
                for presentation_doc in vals['text']:
                    for q_a_doc in vals_2['text']:
                        doc_graph.add_edge(presentation_doc, q_a_doc)

