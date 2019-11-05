

def load_docs(fname, nrows=None):
    docs = {}
    cntr = 0
    with open(fname, 'r') as f:
        for line in f:
            idx, doc = line.split("\t")
            idx = int(idx)
            docs[idx] = doc

            cntr += 1
            if nrows is not None:
                if cntr == nrows:
                    break

    return docs


