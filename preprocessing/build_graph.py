""
from preprocessing.conference_call_analysis_parallel import loopem
import config.config as cfg
from preprocessing.graph import DocGraph

files = os.listdir(cfg.vals["raw_data_dir"])[:10000]
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
    #try:
    doc_graph = loopem(fname, doc_graph)
    success_cntr += 1
    #except (AttributeError, KeyError, IndexError) as err:
    #    pass

    progress = 100 * ((cntr + 1) / n_files)
    print("Progress: {:.2f}%".format(progress), end='\r')


print("{}/{} successfully extracted".format(success_cntr, n_files))

doc_graph.write()
choose = doc_graph.sample_edges(cfg.vals['train_sample_size'])
#doc_graph.gen_train_data(choose)
