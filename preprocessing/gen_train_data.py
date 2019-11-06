import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from preprocessing.graph import DocGraph


doc_graph = DocGraph()
doc_graph.load_docs(cfg.vals["clean_data_dir"] + "doc-list.txt")
doc_graph.load_graph(cfg.vals["clean_data_dir"] + "edge-list.txt")
print("Document and edge lists read from disk")

choose = doc_graph.sample_edges(cfg.vals['train_sample_size'])
doc_graph.gen_train_data(choose)

