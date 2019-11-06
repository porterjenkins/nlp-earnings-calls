import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nltk.corpus import stopwords
import string

class Document(object):
    stop = set(stopwords.words('english') + ['would', 'thing', 'question', 'could', ''])

    def __init__(self, text, speaker=None, speaker_type=None, document_type=None, id=None):
        self.tokens = Document.clean_tokenize_doc(text)
        self.text = text
        self.speaker = speaker
        self.speaker_type = speaker_type
        self.document_type = document_type
        self.id = id

    def __str__(self):
        return self.text

    def set_speaker_type(self, speaker_type):
        self.speaker_type = speaker_type
    def set_speaker(self, speaker):
        self.speaker = speaker

    def set_id(self, id):
        self.id = id

    @classmethod
    def clean_tokenize_doc(cls, doc):
        doc_tokenized = []
        doc = doc.strip().replace("\n", "")
        for word in doc.split():
            word = word.translate(str.maketrans('', '', string.punctuation))
            word = word.lower()
            if word not in Document.stop:
                doc_tokenized.append(word)
        return doc_tokenized
