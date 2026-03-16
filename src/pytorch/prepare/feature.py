import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"\d+","",text)
    text = re.sub(r"[^\w\s]","",text)
    text = re.sub(r"\s+"," ",text).strip()

    return text


def build_vectorizer():

    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1,2),
        min_df=5,
        stop_words="english"
    )


def encode_labels(labels):
    unique = sorted(list(set(labels)))
    mapping = {label:i for i,label in enumerate(unique)}
    encoded = np.array([mapping[l] for l in labels])

    return encoded, mapping
