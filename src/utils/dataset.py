import re
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def normalize_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ").replace("\ufeff", " ")
    return " ".join(text.split()).strip().lower()


def tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def encode_text(text, stoi, seq_len):
    tokens = tokenize(normalize_text(text))
    unk_idx = stoi[UNK_TOKEN]
    ids = [stoi.get(tok, unk_idx) for tok in tokens][:seq_len]
    if len(ids) < seq_len:
        ids += [stoi[PAD_TOKEN]] * (seq_len - len(ids))
    return ids


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for txt in texts:
        counter.update(tokenize(normalize_text(txt)))

    stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            stoi[token] = len(stoi)
    return stoi
