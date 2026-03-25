import pandas as pd
import numpy as np
import ollama
from pathlib import Path
from datasets import load_dataset
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================
# NORMALIZE LABELS
# =========================
def normalize_label(label):
    label = str(label).lower()

    if "gpt" in label or "openai" in label:
        return "OpenAI"
    elif "llama" in label or "meta" in label:
        return "Meta"
    elif "gemini" in label or "google" in label:
        return "Google"
    elif "mistral" in label:
        return "Mistral"
    elif "human" in label:
        return "Human"
    else:
        return None

project_root = Path(__file__).resolve().parents[1]
# =========================
# DATASETS
# =========================
def get_prof_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset_path = project_root / "data" / "dataset-exemplos.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df["id"] = df["ID"]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(
        min(n_lines, len(df)), random_state=42
    ).reset_index(drop=True)


def get_subm1_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset_path = project_root / "data" / "subm1_labels_revealed.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    required = {"Text", "Label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns {required}. Found columns: {list(df.columns)}"
        )

    return df[["Text", "Label"]].dropna().sample(
        min(n_lines, len(df)), random_state=42
    ).reset_index(drop=True)


def get_otb_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    df = pd.concat([df_train, df_test], ignore_index=True)

    mapping_classes = {
        "meta-llama": "Meta",
        "qwen": "OpenAI",
        "google": "Google",
        "anthropic": "Anthropic",
    }

    df["id"] = df["url"]
    df["Text"] = df["content"]
    df["Label"] = df["model"].apply(lambda x: mapping_classes.get(x.split("/")[0].lower(), "Others"))
    df = df[df["Label"] != "Others"]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(
        min(n_lines, len(df)), random_state=42
    ).reset_index(drop=True)


def get_atdp_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset = load_dataset("artem9k/ai-text-detection-pile")

    df = dataset["train"].to_pandas()

    df["Text"] = df["text"]
    df = df[df["source"] == "human"]
    df["Label"] = df["source"].apply(lambda x: "Human" if x == "human" else "")
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(
        min(n_lines, len(df)), random_state=42
    ).reset_index(drop=True)


def get_ap_dataset(n_lines: int = 3000) -> pd.DataFrame:
    dataset = load_dataset("Anthropic/persuasion")

    df = dataset["train"].to_pandas()

    df["id"] = df["worker_id"]
    df["Text"] = df["argument"]
    df["Label"] = df["source"].apply(lambda x: "Anthropic" if x.startswith("Claude") else "Human")
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(
        min(n_lines, len(df)), random_state=42
    ).reset_index(drop=True)

# =========================
# COMBINE + BALANCE
# =========================
def get_combined_dataset(n_lines=30000):
    dfs = [
        get_prof_dataset(n_lines // 3),
        get_subm1_dataset(n_lines // 3),
        get_otb_dataset(n_lines // 3),
        get_atdp_dataset(n_lines // 3),
        get_ap_dataset(n_lines // 3)
    ]

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna().drop_duplicates(subset=["Text"])

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def balance_dataset(df, max_per_class=3000):
    return pd.concat([
        df[df["Label"] == l].sample(min(len(df[df["Label"] == l]), max_per_class), random_state=42)
        for l in df["Label"].unique()
    ]).reset_index(drop=True)


# =========================
# RAG (TF-IDF)
# =========================
def build_retriever(df):
    texts = df["Text"].tolist()
    labels = df["Label"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(texts)

    return vectorizer, X, texts, labels


def retrieve_examples(query, vectorizer, X, texts, labels, k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X)[0]
    idx = sims.argsort()[-k:][::-1]

    return [(texts[i], labels[i]) for i in idx]


# =========================
# PROMPT
# =========================
def build_prompt(text, examples):
    examples_str = ""
    for t, l in examples:
        examples_str += f'Text: "{t[:200]}"\nLabel: {l}\n\n'

    return f"""
Classify the text into:
OpenAI, Meta, Google, Mistral, Human

Examples:
{examples_str}

Text: "{text}"

Answer ONLY with the label.
"""


# =========================
# OLLAMA
# =========================
def classify_with_ollama(prompt, model="mistral"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


def clean_label(label):
    classes = ["OpenAI", "Meta", "Google", "Mistral", "Human"]
    for c in classes:
        if c.lower() in label.lower():
            return c
    return "Human"


# =========================
# EVALUATION
# =========================
def evaluate(train_df, val_df):
    print("🔍 Building retriever...")
    vectorizer, X, texts, labels = build_retriever(train_df)

    y_true, y_pred = [], []

    print("Evaluating...")

    #for i, row in val_df.iterrows():
    for i, (_, row) in enumerate(val_df.iterrows()):
        text = row["Text"]

        examples = retrieve_examples(text, vectorizer, X, texts, labels)
        prompt = build_prompt(text, examples)

        try:
            pred = clean_label(classify_with_ollama(prompt))
        except:
            pred = "Human"

        y_true.append(row["Label"])
        y_pred.append(pred)

        if i % 10 == 0:
            print(f"{i}/{len(val_df)}")

    print("\n RESULTS")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    return vectorizer, X, texts, labels


# =========================
# SUBMISSION
# =========================
def generate_submission(vectorizer, X, texts, labels):

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "data" / "subm2.csv"
    print(model_path)


    test_df = pd.read_csv(model_path, sep=";")

    preds = []

    print("Generating submission...")

    for i, row in test_df.iterrows():
        text = row["Text"]

        examples = retrieve_examples(text, vectorizer, X, texts, labels)
        prompt = build_prompt(text, examples)

        try:
            label = clean_label(classify_with_ollama(prompt))
        except:
            label = "Human"

        preds.append(label)

        if i % 10 == 0:
            print(f"{i}/{len(test_df)}")

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "Text": test_df["Text"],
        "Label": preds
    })

    model_path_write = project_root / "data" / "subm2_pred2.csv"
    submission.to_csv(model_path_write, index=False, quoting=csv.QUOTE_NONE, sep=";", escapechar="\\")
    print("subm2_pred2.csv criado!")


# =========================
# MAIN
# =========================
def main():
    print("Loading datasets...")
    df = balance_dataset(get_combined_dataset(30000))

    print(df["Label"].value_counts())

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    vectorizer, X, texts, labels = evaluate(train_df, val_df)

    generate_submission(vectorizer, X, texts, labels)


if __name__ == "__main__":
    main()