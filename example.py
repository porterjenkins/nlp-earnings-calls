from utils import load_doc_embedding, load_docs


# change the filepath to reference your local file
path_to_embedding = "/Volumes/Porter's Data/penn-state/data-sets/nlp-earnings-calls/output/starspace_docs.txt"
path_to_docs = "/Volumes/Porter's Data/penn-state/data-sets/nlp-earnings-calls/clean/raw-doc-list.txt"
limit = 10


# read embedding vectors from disk
# nrows limits to the k rows; just for quick loading
embedding = load_doc_embedding(path_to_embedding, nrows=limit)
documents =  load_docs(path_to_docs, nrows=limit)

for doc_id, embed_vec in embedding.items():
    raw_text = documents.get(doc_id, None)
    if raw_text is None:
        continue

    print(f"Document {doc_id}: {raw_text} {embed_vec}")