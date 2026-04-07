from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from collections import Counter
from collections.abc import Iterable

from models.dnn.dnnutils.vocab import Vocab


class TFIDF:
    def __init__(self, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 1), max_features: int | None = None):
        if analyzer not in {"word", "char"}:
            raise ValueError("analyzer must be 'word' or 'char'")
        if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must contain valid positive bounds")

        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab = Vocab(unk_token=None)
        self.idf: np.ndarray | None = None

    def _ngrams(self, text: str) -> list[str]:
        min_n, max_n = self.ngram_range

        if self.analyzer == "word":
            tokens = text.split()
            ngrams: list[str] = []
            for n in range(min_n, max_n + 1):
                if len(tokens) < n:
                    continue
                ngrams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
            return ngrams

        normalized = " ".join(text.split())
        ngrams = []
        for n in range(min_n, max_n + 1):
            if len(normalized) < n:
                continue
            ngrams.extend(normalized[i : i + n] for i in range(len(normalized) - n + 1))
        return ngrams

    def fit(self, corpus: Iterable[str]) -> "TFIDF":
        documents = list(corpus)
        df_counter: Counter[str] = Counter()

        for doc in documents:
            df_counter.update(set(self._ngrams(doc)))

        if self.max_features is not None:
            sorted_terms = sorted(df_counter.items(), key=lambda item: (-item[1], item[0]))[: self.max_features]
            terms = [term for term, _ in sorted_terms]
        else:
            terms = sorted(df_counter)

        self.vocab = Vocab(tokens=terms, unk_token=None)
        self.vocab.frequencies.update(df_counter)

        n_docs = len(documents)
        df = np.zeros(len(self.vocab), dtype=np.float32)

        for doc in documents:
            for term in set(self._ngrams(doc)):
                if term in self.vocab:
                    df[self.vocab[term]] += np.float32(1.0)

        self.idf = (np.log((1 + n_docs) / (1 + df)) + 1.0).astype(np.float32)
        return self

    def transform(self, corpus: Iterable[str]) -> csr_matrix:
        """Returns a sparse CSR matrix (n_docs × vocab_size) to avoid OOM on large corpora."""
        if self.idf is None:
            raise ValueError("TFIDF must be fitted before calling transform")

        documents = list(corpus)
        n_docs = len(documents)
        n_vocab = len(self.vocab)

        rows, cols, data = [], [], []
        for row_index, doc in enumerate(documents):
            terms = self._ngrams(doc)
            if not terms:
                continue
            term_counts: Counter[str] = Counter(terms)
            n_terms_in_doc = len(terms)
            for term, count in term_counts.items():
                if term in self.vocab:
                    rows.append(row_index)
                    cols.append(self.vocab[term])
                    data.append(np.float32(count) / np.float32(n_terms_in_doc))

        if not data:
            return csr_matrix((n_docs, n_vocab), dtype=np.float32)

        tf = csr_matrix((data, (rows, cols)), shape=(n_docs, n_vocab), dtype=np.float32)

        # Apply IDF (element-wise per column)
        tfidf = tf.multiply(self.idf)

        # L2-normalise each row without creating a dense intermediate
        norms = np.asarray(tfidf.power(2).sum(axis=1)).ravel()
        norms[norms == 0] = 1.0
        row_scale = diags(1.0 / np.sqrt(norms), dtype=np.float32)
        return (row_scale @ tfidf).astype(np.float32)

    def fit_transform(self, corpus: Iterable[str]) -> csr_matrix:
        documents = list(corpus)
        self.fit(documents)
        return self.transform(documents)

    @property
    def token_to_index(self) -> dict[str, int]:
        return self.vocab.token_to_index
