import re
import numpy as np
from collections import Counter


stop_words = {
    "a", "an", "the", "and", "or", "but", "if", "while",
    "with", "without", "in", "on", "at", "by", "for", "to", "from",
}


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


def build_text_vector(text, tfidf_word, tfidf_char, hand_mean, hand_std, hand_feature_names):
    raw_text = str(text)
    clean_text = preprocess_text(raw_text)

    x_word = tfidf_word.transform([clean_text])
    x_char = tfidf_char.transform([clean_text])

    feat = extract_features(raw_text, clean_text)
    x_hand = np.array([[feat[name] for name in hand_feature_names]], dtype=float)
    x_hand = (x_hand - hand_mean) / hand_std

    return np.hstack([x_word, x_char, x_hand])
