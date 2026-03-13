import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


class TFIDF:
    def __init__(self, analyzer="word", ngram_range=(1, 1), max_features=None):
        if analyzer not in {"word", "char"}:
            raise ValueError("analyzer must be 'word' or 'char'")

        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab = {}
        self.idf = None

    def _ngrams(self, text):
        min_n, max_n = self.ngram_range

        if self.analyzer == "word":
            tokens = text.split()
            ngrams = []
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

    def fit(self, corpus: list[str]):
        # Build vocabulary and compute document frequencies
        df_counter = Counter()

        for doc in corpus:
            doc_terms = set(self._ngrams(doc))
            df_counter.update(doc_terms)

        if self.max_features is not None:
            sorted_terms = sorted(df_counter.items(), key=lambda x: (-x[1], x[0]))[: self.max_features]
            terms = [term for term, _ in sorted_terms]
        else:
            terms = sorted(df_counter.keys())

        self.vocab = {term: i for i, term in enumerate(terms)}
        n_docs = len(corpus)
        n_terms = len(self.vocab)

        # Compute IDF
        df = np.zeros(n_terms)
        for doc in corpus:
            terms_in_doc = set(self._ngrams(doc))
            for term in terms_in_doc:
                if term in self.vocab:
                    df[self.vocab[term]] += 1

        self.idf = np.log((1 + n_docs) / (1 + df)) + 1  # smooth IDF

    def transform(self, corpus: list[str]) -> np.ndarray:
        n_docs = len(corpus)
        n_terms = len(self.vocab)
        tf = np.zeros((n_docs, n_terms))

        for i, doc in enumerate(corpus):
            terms = self._ngrams(doc)
            for term in terms:
                if term in self.vocab:
                    tf[i, self.vocab[term]] += 1
            if len(terms) > 0:
                tf[i] /= len(terms)  # normalize by number of ngrams

        tfidf = tf * self.idf

        # L2 normalize each row
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return tfidf / norms

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)


def remove_stop_words(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


def extract_features(raw_text, clean_text):
    words = clean_text.split()
    num_words = len(words)
    raw_lower = raw_text.lower()

    sentence_split = raw_text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentence_split if s.strip()]
    sentence_lengths = [len(preprocess_text(s).split()) for s in sentences]

    word_counts = Counter(words)
    punctuation = {p: raw_text.count(p) for p in [",", ".", ";", ":", "!", "?"]}
    n_chars = len(raw_text) if raw_text else 1

    features = {}

    # --- Basic stats ---
    features["length_chars"] = len(raw_text)
    features["num_words"] = num_words
    features["num_sentences"] = len(sentences)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if num_words > 0 else 0.0
    features["avg_sentence_length"] = np.mean(sentence_lengths) if sentence_lengths else 0.0
    features["sentence_length_std"] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    # --- Lexical diversity ---
    features["type_token_ratio"] = len(set(words)) / num_words if num_words > 0 else 0.0
    features["hapax_ratio"] = sum(1 for c in word_counts.values() if c == 1) / num_words if num_words > 0 else 0.0
    features["repetition_ratio"] = max(word_counts.values()) / num_words if num_words > 0 else 0.0

    # --- Punctuation ratios ---
    for p, count in punctuation.items():
        key = f"punct_{p}_ratio".replace("?", "q").replace("!", "e").replace(".", "dot")
        features[key] = count / n_chars

    # --- Char-class ratios ---
    original_tokens = [token for token in preprocess_text(raw_lower).split() if token]
    features["stopword_ratio"] = (
        sum(1 for token in original_tokens if token in stop_words) / len(original_tokens)
        if original_tokens else 0.0
    )
    features["uppercase_ratio"] = sum(1 for ch in raw_text if ch.isupper()) / n_chars
    features["digit_ratio"] = sum(1 for ch in raw_text if ch.isdigit()) / n_chars
    features["whitespace_ratio"] = sum(1 for ch in raw_text if ch.isspace()) / n_chars

    # --- Punctuation problems / informal style (human signals) ---
    # Contractions: don't, I'm, can't, etc.
    import re
    n_contractions = len(re.findall(r"\b\w+'\w+\b", raw_text))
    features["contraction_ratio"] = n_contractions / num_words if num_words > 0 else 0.0

    # Repeated characters: loool, nooo, yesss (informal/expressive)
    features["char_repeat_count"] = len(re.findall(r"(.)\1{2,}", raw_text)) / n_chars

    # All-caps words: GREAT, WOW, NO (human emphasis)
    all_caps_words = [w for w in raw_text.split() if len(w) > 1 and w.isupper()]
    features["allcaps_word_ratio"] = len(all_caps_words) / num_words if num_words > 0 else 0.0

    # Multiple punctuation: !!, ??, !? (informal)
    features["multi_punct_count"] = len(re.findall(r"[!?]{2,}", raw_text)) / n_chars

    # Ellipsis usage: ... (human trailing thought)
    features["ellipsis_count"] = raw_text.count("...") / n_chars

    # Missing space after punctuation: "hello.World" (typo/informal)
    features["missing_space_after_punct"] = len(re.findall(r"[.!?,;:][A-Za-z]", raw_text)) / n_chars

    # Sentence fragments: sentences with 1-3 words (human casual)
    features["fragment_ratio"] = (
        sum(1 for sl in sentence_lengths if sl <= 3) / len(sentence_lengths)
        if sentence_lengths else 0.0
    )

    # Run-on sentences: sentences with 40+ words (human stream-of-consciousness)
    features["runon_ratio"] = (
        sum(1 for sl in sentence_lengths if sl >= 40) / len(sentence_lengths)
        if sentence_lengths else 0.0
    )

    # Informal filler words (human verbal habit)
    filler_words = {"tbh", "lol", "omg", "idk", "imo", "btw", "ngl", "smh", "fyi",
                    "like", "just", "really", "basically", "literally", "actually",
                    "kinda", "sorta", "yeah", "yep", "nope", "ok", "okay"}
    raw_word_tokens = re.findall(r"\b\w+\b", raw_lower)
    features["filler_word_ratio"] = (
        sum(1 for w in raw_word_tokens if w in filler_words) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # AI formal connector words (AI signal — lower = more human)
    ai_connectors = {"furthermore", "moreover", "additionally", "consequently", "nevertheless",
                     "therefore", "thus", "hence", "notwithstanding", "whereby",
                     "subsequently", "henceforth", "aforementioned"}
    features["ai_connector_ratio"] = (
        sum(1 for w in raw_word_tokens if w in ai_connectors) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # First-person pronoun density (human writing is more personal)
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    features["first_person_ratio"] = (
        sum(1 for w in raw_word_tokens if w in first_person) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # Question density (human writing asks more questions)
    features["question_density"] = raw_text.count("?") / len(sentences) if sentences else 0.0

    # Exclamation density
    features["exclamation_density"] = raw_text.count("!") / len(sentences) if sentences else 0.0

    return features


def preprocess_text(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    return text


def build_handcrafted_matrix(raw_texts, clean_texts):
    feature_dicts = [extract_features(raw_text, clean_text) for raw_text, clean_text in zip(raw_texts, clean_texts)]
    feature_names = sorted(feature_dicts[0].keys())
    matrix = np.array([[feat[name] for name in feature_names] for feat in feature_dicts], dtype=float)
    return matrix, feature_names


def standardize_train_test(train_matrix, test_matrix):
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1
    train_scaled = (train_matrix - mean) / std
    test_scaled = (test_matrix - mean) / std
    return train_scaled, test_scaled, mean, std

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Maintains class proportions in train and test sets.
    """
    np.random.seed(random_state)

    # Get unique classes and their indices
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    # Split each class separately
    for class_label in unique_classes:
        # Find all indices for this class
        class_indices = np.where(y == class_label)[0]
        n_samples = len(class_indices)

        # Shuffle indices
        np.random.shuffle(class_indices)

        # Calculate split point
        split_point = int(n_samples * (1 - test_size))

        # Add to train/test
        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])

    # Shuffle final indices
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def classification_report_from_cm(cm, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nPer-class metrics:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")

    for idx, class_name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"{class_name:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")


def build_text_vector(text, tfidf_word, tfidf_char, hand_mean, hand_std, feature_names):
    raw_text = str(text)
    clean_text = preprocess_text(raw_text)

    x_word = tfidf_word.transform([clean_text])
    x_char = tfidf_char.transform([clean_text])

    feat = extract_features(raw_text, clean_text)
    x_hand = np.array([[feat[name] for name in feature_names]], dtype=float)
    x_hand = (x_hand - hand_mean) / hand_std

    return np.hstack([x_word, x_char, x_hand])


def predict_text(text, model, tfidf_word, tfidf_char, hand_mean, hand_std, feature_names, class_names):
    x = build_text_vector(text, tfidf_word, tfidf_char, hand_mean, hand_std, feature_names)
    probs = model.predict(x, training=False)[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], probs


stop_words = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "with",
        "without",
        "in",
        "on",
        "at",
        "by",
        "for",
        "to",
        "from",
    ]
)


project_root = Path(__file__).resolve().parent
dataset_path = project_root / "data" / "dataset-exemplos.csv"

df = pd.read_csv(dataset_path, sep=";")

dataset_path = project_root / "data" / "dataset-hf.csv"

df = pd.concat([df, pd.read_csv(dataset_path, sep=",")], ignore_index=True)

df.columns = [c.strip().capitalize() for c in df.columns]
df = df[["Text", "Label"]]

df["Text"] = df["Text"].astype(str)
df["Text_clean"] = df["Text"].apply(preprocess_text)

class_names = sorted(df["Label"].unique())
label_to_num = {label: i for i, label in enumerate(class_names)}
df["Label_num"] = df["Label"].map(label_to_num)

y = df["Label_num"].values
indices = np.arange(len(df))
idx_train, idx_test, _, _ = train_test_split(indices, y, test_size=0.2, random_state=42)

df_train = df.iloc[idx_train].reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)

y_train = df_train["Label_num"].values
y_test = df_test["Label_num"].values

tfidf_word = TFIDF(analyzer="word", ngram_range=(1, 2), max_features=4000)
tfidf_char = TFIDF(analyzer="char", ngram_range=(3, 5), max_features=3000)

X_word_train = tfidf_word.fit_transform(df_train["Text_clean"].tolist())
X_word_test = tfidf_word.transform(df_test["Text_clean"].tolist())

X_char_train = tfidf_char.fit_transform(df_train["Text_clean"].tolist())
X_char_test = tfidf_char.transform(df_test["Text_clean"].tolist())

X_hand_train, hand_feature_names = build_handcrafted_matrix(df_train["Text"].tolist(), df_train["Text_clean"].tolist())
X_hand_test, _ = build_handcrafted_matrix(df_test["Text"].tolist(), df_test["Text_clean"].tolist())
X_hand_train, X_hand_test, hand_mean, hand_std = standardize_train_test(X_hand_train, X_hand_test)

X_train = np.hstack([X_word_train, X_char_train, X_hand_train])
X_test = np.hstack([X_word_test, X_char_test, X_hand_test])

print("Word TF-IDF train shape:", X_word_train.shape)
print("Char TF-IDF train shape:", X_char_train.shape)
print("Handcrafted features shape:", X_hand_train.shape)
print("Final X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

print("y_train class distribution:", np.bincount(y_train) / len(y_train))
print("y_test class distribution:", np.bincount(y_test) / len(y_test))


from dnn.model import nn

history = nn.fit(
    X_train,
    y_train,
    epochs=300,
    learning_rate=0.01,
    batch_size=16,
    x_val=X_test,
    y_val=y_test,
    verbose_every=25,
    patience=30,
    min_delta=1e-4,
    lr_decay=0.5,
    lr_patience=10,
)

predictions = nn.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == y_test)
print("Predicted label distribution:", np.bincount(predicted_labels, minlength=len(class_names)))
print("True label distribution:", np.bincount(y_test, minlength=len(class_names)))
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Best Val Loss: {min(history['val_loss']):.4f}")

cm = confusion_matrix(y_test, predicted_labels, num_classes=len(class_names))
classification_report_from_cm(cm, class_names)


sample_text = "I'm human tbh, I can't understand tbh, not really, what is going on my friend!!"
sample_label, sample_probs = predict_text(
    sample_text,
    nn,
    tfidf_word,
    tfidf_char,
    hand_mean,
    hand_std,
    hand_feature_names,
    class_names,
)
print("\nSingle text prediction:")
print("Text:", sample_text)
print("Predicted class:", sample_label)
print("Class probabilities:", np.round(sample_probs, 4))
