import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_docs


docs = load_docs(cfg.vals["clean_data_dir"] + "doc-list.txt", 100)
documents = [TaggedDocument(doc, [i]) for i, doc in docs.items()]
print("training model...")
model = Doc2Vec(documents, vector_size=cfg.vals['hidden_size'], window=2, min_count=1, workers=4)


print("writing output...")
with open(cfg.vals["output_dir"] + "doc2vec.txt", 'w') as f:

    for d in documents:
        idx = d[1][0]
        vec = model.docvecs[idx]
        f.write("{}\t".format(idx))
        for component in vec:
            f.write("{} ".format(component))
        f.write("\n")

print("done")