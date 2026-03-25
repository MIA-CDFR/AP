import re
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


stop_words = {
    "a", "an", "the", "and", "or", "but", "if", "while",
    "with", "without", "in", "on", "at", "by", "for", "to", "from",
}


def preprocess_text(text):
    """Aggressive cleaning used for TF-IDF vectorizer."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_clean(text):
    """Light cleaning for handcrafted feature extraction (keeps punctuation info)."""
    text = str(text).lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    return text


def extract_features(raw_text, clean_text):
    """Extract handcrafted features from raw and lightly-cleaned text."""
    words = clean_text.split()
    num_words = len(words)
    raw_lower = raw_text.lower()

    sentence_split = raw_text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentence_split if s.strip()]
    sentence_lengths = [len(preprocess_text_clean(s).split()) for s in sentences]

    word_counts = Counter(words)
    # Only keep punctuation marks with actual signal (! and ? are ~0 in scientific text)
    punctuation = {p: raw_text.count(p) for p in [",", ".", ";", ":"]}
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
        key = f"punct_{p}_ratio".replace(".", "dot")
        features[key] = count / n_chars

    # --- Char-class ratios ---
    original_tokens = [token for token in preprocess_text_clean(raw_lower).split() if token]
    features["stopword_ratio"] = (
        sum(1 for token in original_tokens if token in stop_words) / len(original_tokens)
        if original_tokens else 0.0
    )
    features["uppercase_ratio"] = sum(1 for ch in raw_text if ch.isupper()) / n_chars
    features["digit_ratio"] = sum(1 for ch in raw_text if ch.isdigit()) / n_chars
    features["whitespace_ratio"] = sum(1 for ch in raw_text if ch.isspace()) / n_chars

    # --- Informal / style signals ---
    n_contractions = len(re.findall(r"\b\w+'\w+\b", raw_text))
    features["contraction_ratio"] = n_contractions / num_words if num_words > 0 else 0.0

    # All-caps words: acronyms like DNA, PFAS — higher in human scientific papers
    all_caps_words = [w for w in raw_text.split() if len(w) > 1 and w.isupper()]
    features["allcaps_word_ratio"] = len(all_caps_words) / num_words if num_words > 0 else 0.0

    # Missing space after punctuation (e.g. inline formula references)
    features["missing_space_after_punct"] = len(re.findall(r"[.!?,;:][A-Za-z]", raw_text)) / n_chars

    # Sentence fragments: sentences with 1-3 words (human casual writing)
    features["fragment_ratio"] = (
        sum(1 for sl in sentence_lengths if sl <= 3) / len(sentence_lengths)
        if sentence_lengths else 0.0
    )

    # Run-on sentences: 40+ words (more common in human academic papers)
    features["runon_ratio"] = (
        sum(1 for sl in sentence_lengths if sl >= 40) / len(sentence_lengths)
        if sentence_lengths else 0.0
    )

    raw_word_tokens = re.findall(r"\b\w+\b", raw_lower)

    # Informal filler words (human verbal habit)
    filler_words = {"tbh", "lol", "omg", "idk", "imo", "btw", "ngl", "smh", "fyi",
                    "like", "just", "really", "basically", "literally", "actually",
                    "kinda", "sorta", "yeah", "yep", "nope", "ok", "okay"}
    features["filler_word_ratio"] = (
        sum(1 for w in raw_word_tokens if w in filler_words) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # AI formal connector words
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

    # Question density (rare in scientific text but discriminates Human from AI)
    features["question_density"] = raw_text.count("?") / len(sentences) if sentences else 0.0

    # --- AI-specific linguistic signals ---

    # Bigram uniqueness (AI tends to repeat key concepts in a more structured way)
    if len(words) > 1:
        bigrams = list(zip(words[:-1], words[1:]))
        features["bigram_uniqueness"] = len(set(bigrams)) / len(bigrams)
    else:
        features["bigram_uniqueness"] = 0.0

    # Adverb overuse (AI overuses intensifiers)
    adverbs = {"very", "quite", "extremely", "remarkably", "clearly", "obviously",
               "certainly", "absolutely", "definitely", "highly", "particularly",
               "significantly", "substantially", "considerably"}
    features["adverb_ratio"] = (
        sum(1 for w in raw_word_tokens if w in adverbs) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # Modal verbs (AI hedging language)
    modals = {"would", "could", "should", "might", "may", "must", "can", "will", "shall"}
    features["modal_ratio"] = (
        sum(1 for w in raw_word_tokens if w in modals) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # Discourse connectors (true connectors only, no stopword contamination)
    discourse_connectors = {"however", "therefore", "furthermore", "moreover", "consequently",
                            "conversely", "nonetheless", "nevertheless", "alternatively",
                            "notwithstanding", "thus", "hence", "accordingly", "subsequently"}
    features["transition_ratio"] = (
        sum(1 for w in raw_word_tokens if w in discourse_connectors) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    # Passive voice ratio (AI uses more passive constructions)
    be_verbs = {"is", "was", "are", "were", "be", "been", "being"}
    passive_constructions = sum(1 for w in raw_word_tokens if w in be_verbs)
    features["passive_voice_ratio"] = passive_constructions / len(sentences) if sentences else 0.0

    # --- Domain-specific signals (scientific text) ---

    # Citation pattern: human scientific papers cite sources inline
    citation_patterns = len(re.findall(r"\b\w+,\s*\d{4}\b|\bet\s+al\b", raw_text))
    features["citation_density"] = citation_patterns / len(sentences) if sentences else 0.0

    # Hedging language: differs between human papers and AI summaries
    hedging_words = {"suggest", "suggests", "suggested", "appears", "appear", "indicates",
                     "indicate", "hypothesize", "hypothesizes", "propose", "proposes",
                     "believed", "thought", "considered", "estimated", "likely", "unlikely",
                     "possibly", "perhaps", "presumably", "seemingly", "arguably"}
    features["hedging_ratio"] = (
        sum(1 for w in raw_word_tokens if w in hedging_words) / len(raw_word_tokens)
        if raw_word_tokens else 0.0
    )

    return features


def build_handcrafted_matrix(raw_texts, clean_texts):
    feature_dicts = [extract_features(r, c) for r, c in zip(raw_texts, clean_texts)]
    feature_names = sorted(feature_dicts[0].keys())
    matrix = np.array([[fd[name] for name in feature_names] for fd in feature_dicts], dtype=float)
    return matrix, feature_names


def standardize_train_test(train_matrix, test_matrix):
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return (train_matrix - mean) / std, (test_matrix - mean) / std, mean, std


def build_vectorizer():
    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=5,
        # stop_words="english"
    )


def encode_labels(labels):
    unique = sorted(list(set(labels)))
    mapping = {label: i for i, label in enumerate(unique)}
    encoded = np.array([mapping[l] for l in labels])
    return encoded, mapping

