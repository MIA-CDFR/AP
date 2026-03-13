import numpy as np
import pandas as pd

from datasets import load_dataset
from pathlib import Path

from dnn.prepare.feature import (
    preprocess_text,
    build_handcrafted_matrix,
    standardize_train_test,
    build_text_vector,
)
from dnn.prepare.tf_idf import TFIDF


def get_prof_dataset(n_lines: int = 10000) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset-exemplos.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df["id"] = df["ID"]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_otb_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    df = pd.concat([df_train, df_test], ignore_index=True)

    mapping_classes = {
        "meta-llama": "Meta",
        "qwen": "OpenAI",
        "mistralai": "Mistral",
        "google": "Google",
        "anthropic": "Anthropic",
    }

    df["id"] = df["url"]
    df["Text"] = df["content"]
    df["Label"] = df["model"].apply(lambda x: mapping_classes.get(x.split("/")[0].lower(), "Others"))
    df = df[df["Label"] != "Others"]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_atdp_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset = load_dataset("artem9k/ai-text-detection-pile")

    df = dataset["train"].to_pandas()

    df["Text"] = df["text"]
    df = df[df["source"] == "human"]
    df["Label"] = df["source"].apply(lambda x: "Human" if x == "human" else "")
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_ap_dataset(n_lines: int = 3000) -> pd.DataFrame:
    dataset = load_dataset("Anthropic/persuasion")

    df = dataset["train"].to_pandas()

    df["id"] = df["worker_id"]
    df["Text"] = df["argument"]
    df["Label"] = df["source"].apply(lambda x: "Anthropic" if x.startswith("Claude") else "Human")
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_datasets() -> pd.DataFrame:
    df_prof = get_prof_dataset()
    df_otb = get_otb_dataset()
    df_atdp = get_atdp_dataset()
    df_ap = get_ap_dataset()

    df = pd.concat([df_prof, df_otb, df_atdp, df_ap], ignore_index=True)

    return df


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


class DatasetLoader:
    def prepare_text(self, text: str) -> np.ndarray:
        return build_text_vector(
            text,
            self.tfidf_word,
            self.tfidf_char,
            self.hand_mean,
            self.hand_std,
            self.hand_feature_names,
        )

    @classmethod
    def load_datasets(cls) -> "DatasetLoader":
        df = get_datasets()

        df.columns = [c.strip().capitalize() for c in df.columns]
        df = df[["Text", "Label"]]

        df["Text"] = df["Text"].astype(str)
        df["Text_clean"] = df["Text"].apply(preprocess_text)

        cls.class_names = sorted(df["Label"].unique())
        label_to_num = {label: i for i, label in enumerate(cls.class_names)}
        df["Label_num"] = df["Label"].map(label_to_num)

        y = df["Label_num"].values
        indices = np.arange(len(df))
        idx_train, idx_test, _, _ = train_test_split(indices, y, test_size=0.2, random_state=42)

        df_train = df.iloc[idx_train].reset_index(drop=True)
        df_test = df.iloc[idx_test].reset_index(drop=True)

        cls.y_train = df_train["Label_num"].values
        cls.y_test = df_test["Label_num"].values

        cls.tfidf_word = TFIDF(analyzer="word", ngram_range=(1, 2), max_features=5000)
        cls.tfidf_char = TFIDF(analyzer="char", ngram_range=(3, 5), max_features=5000)

        X_word_train = cls.tfidf_word.fit_transform(df_train["Text_clean"].tolist())
        X_word_test = cls.tfidf_word.transform(df_test["Text_clean"].tolist())

        X_char_train = cls.tfidf_char.fit_transform(df_train["Text_clean"].tolist())
        X_char_test = cls.tfidf_char.transform(df_test["Text_clean"].tolist())

        X_hand_train, cls.hand_feature_names = build_handcrafted_matrix(
            df_train["Text"].tolist(), df_train["Text_clean"].tolist()
        )
        X_hand_test, _ = build_handcrafted_matrix(
            df_test["Text"].tolist(), df_test["Text_clean"].tolist()
        )
        cls.X_hand_train, cls.X_hand_test, cls.hand_mean, cls.hand_std = standardize_train_test(
            X_hand_train, X_hand_test
        )

        cls.X_train = np.hstack([X_word_train, X_char_train, cls.X_hand_train]).astype(np.float32)
        cls.X_test = np.hstack([X_word_test, X_char_test, cls.X_hand_test]).astype(np.float32)

        print("Word TF-IDF train shape:", X_word_train.shape)
        print("Char TF-IDF train shape:", X_char_train.shape)
        print("Handcrafted features shape:", X_hand_train.shape)
        print("Final X_train shape:", cls.X_train.shape)
        print("Final X_test shape:", cls.X_test.shape)
        print("y_train class distribution:", np.bincount(cls.y_train) / len(cls.y_train))
        print("y_test class distribution:", np.bincount(cls.y_test) / len(cls.y_test))

        return cls
