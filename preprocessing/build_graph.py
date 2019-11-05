import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.conference_call_analysis_parallel import loopem
import config.config as cfg
from preprocessing.graph import DocGraph

files = os.listdir(cfg.vals["raw_data_dir"])[:500]
n_files = len(files)
success_cntr = 0
docs = {}
edge_list = {}
doc_cntr = 0
doc_graph = DocGraph()

for cntr, f in enumerate(files):
    fname = cfg.vals["raw_data_dir"] + f
    if f == '.DS_Store':
        continue
    try:
        prepared_remarks, managers_qa, analysts_qa, data = loopem(fname, doc_graph)
        success_cntr += 1
    except (AttributeError, KeyError, IndexError) as err:
        pass

    progress = 100 * ((cntr + 1) / n_files)
    print("Progress: {:.4f}".format(progress), end='\r')


print("{}/{} successfully extracted".format(success_cntr, n_files))

doc_graph.write()
