import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg



class DocGraph(object):

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
                f.write("{}\t{}\n".format(idx, text))


        with open(cfg.vals["clean_data_dir"] + "edge-list.txt", 'w') as f:
            for edge in self.edge_list:
                f.write("{}, {}\n".format(edge[0], edge[1]))




    def add_edge(self, doc_1, doc_2):
        doc_1_id = self.docs[doc_1]
        doc_2_id = self.docs[doc_2]

        self.edge_list.add((doc_1_id, doc_2_id))


    @classmethod
    def build_graph(cls, doc_graph, presentations, q_a):

        for speaker, vals in presentations.items():
            for speaker_2, vals_2 in q_a.items():
                for presentation_doc in vals['text']:
                    for q_a_doc in vals_2['text']:
                        doc_graph.add_edge(presentation_doc, q_a_doc)

