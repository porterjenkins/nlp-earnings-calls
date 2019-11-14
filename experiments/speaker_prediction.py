import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_doc_embedding
import config.config as cfg
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve, average_precision_score
import pandas as pd
import matplotlib.pyplot as plt


RANDOM_SEED = 1234

def get_train_test(df, train, test):
    train_out = pd.merge(train['y'], df, how='inner', left_index=True, right_index=True)
    test_out = pd.merge(test['y'], df, how='inner', left_index=True, right_index=True)
    X_train = train_out.iloc[:, 1:]
    X_test = test_out.iloc[:, 1:]
    y_train = train_out['y']
    y_test = test_out['y']

    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_hat, y_hat_probs):
    acc = np.mean(y_hat == y_test.values)
    auc_roc = roc_auc_score(y_true, y_hat_probs)
    f1 = f1_score(y_true, y_hat)
    precision = precision_score(y_true, y_hat)
    recall = recall_score(y_true, y_hat)
    fpr, tpr, _ = roc_curve(y_true, y_hat_probs)
    ap = average_precision_score(y_true, y_hat_probs)

    return acc, auc_roc, f1, precision, recall, fpr, tpr, ap










starspace = load_doc_embedding(cfg.vals["output_dir"] + 'starspace_docs.txt')
starspace = pd.DataFrame.from_dict(starspace, orient='index')

doc2vec = load_doc_embedding(cfg.vals["output_dir"] + 'doc2vec.txt')
doc2vec = pd.DataFrame.from_dict(doc2vec, orient='index')

lda = load_doc_embedding(cfg.vals["output_dir"] + 'lda.txt')
lda = pd.DataFrame.from_dict(lda, orient='index')


metadata = pd.read_csv(cfg.vals["clean_data_dir"] + 'doc-meta-data.csv', index_col=0, header=None)
metadata.columns = ['speaker', 'speaker_type']
metadata = metadata[metadata.speaker_type != ' None']

label_encoder = LabelEncoder()
metadata['y'] = label_encoder.fit_transform(metadata.speaker_type)

train, test = train_test_split(metadata, test_size=.2, random_state=RANDOM_SEED)


train_doc2vec = pd.merge(train['y'], doc2vec, how='inner', left_index=True, right_index=True)
train_lda = pd.merge(train['y'], lda, how='inner', left_index=True, right_index=True)
train_starspace = pd.merge(train['y'], starspace, how='inner', left_index=True, right_index=True)

test_doc2vec = pd.merge(test['y'], doc2vec, how='inner', left_index=True, right_index=True)
test_lda = pd.merge(test['y'], lda, how='inner', left_index=True, right_index=True)
test_starspace = pd.merge(test['y'], starspace, how='inner', left_index=True, right_index=True)


print(train_doc2vec.shape)
print(train_lda.shape)
print(train_starspace.shape)


print(test_doc2vec.shape)
print(test_lda.shape)
print(test_starspace.shape)

embed = {"doc2vec": doc2vec,
         "lda": lda,
         "starspace": starspace}

idx = []
results = []
cols = ['acc', 'auc_roc', 'f1', 'precision', 'recall', 'average precision score']

for label, dta in embed.items():
    X_train, y_train, X_test, y_test = get_train_test(dta, train, test)

    logit = LogisticRegression(penalty='l1', C=.1)
    logit.fit(X=X_train, y=y_train)
    y_hat = logit.predict(X_test)
    probs = logit.predict_proba(X_test)
    acc, auc_roc, f1, precision, recall, fpr, tpr, ap = evaluate(y_true=y_test, y_hat=y_hat, y_hat_probs=probs[:, 1])

    idx.append(label)
    results.append([acc, auc_roc, f1, precision, recall, ap])

    plt.plot(fpr, tpr,
             lw=2, label='{} (auc = {:.4f})'.format(label, auc_roc))


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("auc-roc.pdf")

out = pd.DataFrame(results, columns=cols, index=idx)
out.to_csv("speaker-results.csv")

print(out)
