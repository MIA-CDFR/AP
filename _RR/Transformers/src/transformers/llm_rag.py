import pandas as pd
import numpy as np
import ollama
from pathlib import Path
from datasets import load_dataset
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    #project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset-exemplos.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df["id"] = df["ID"]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(
        min(n_lines, len(df)), random_state=42
    ).reset_index(drop=True)


def get_subm1_dataset(n_lines: int = 10000) -> pd.DataFrame:
    #project_root = Path(__file__).resolve().parents[1]
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
# COMBINE DATASETS
# =========================
def get_combined_dataset(n_lines=30000):

    dfs = []

    dfs.append(get_prof_dataset(n_lines // 3))
    dfs.append(get_subm1_dataset(n_lines // 3))
    dfs.append(get_otb_dataset(n_lines // 3))
    dfs.append(get_atdp_dataset(n_lines // 3))
    dfs.append(get_ap_dataset(n_lines // 3))

    df = pd.concat(dfs, ignore_index=True)

    df = df.dropna()
    df = df.drop_duplicates(subset=["Text"])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# =========================
# BALANCE DATASET
# =========================
def balance_dataset(df, max_per_class=3000):
    balanced = []

    for label in df["Label"].unique():
        subset = df[df["Label"] == label]
        subset = subset.sample(min(len(subset), max_per_class), random_state=42)
        balanced.append(subset)

    return pd.concat(balanced).reset_index(drop=True)


# =========================
# BUILD RETRIEVER (RAG)
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

    top_idx = sims.argsort()[-k:][::-1]

    return [(texts[i], labels[i]) for i in top_idx]


# =========================
# PROMPT
# =========================
def build_prompt(text, examples):
    examples_str = ""

    for t, l in examples:
        examples_str += f'Text: "{t[:200]}"\nLabel: {l}\n\n'

    return f"""
Classify the following text into one of these classes:
- OpenAI
- Meta
- Google
- Mistral
- Human

Here are similar examples:

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


# =========================
# CLEAN LABEL
# =========================
def clean_label(label):
    classes = ["OpenAI", "Meta", "Google", "Mistral", "Human"]

    for c in classes:
        if c.lower() in label.lower():
            return c

    return "Human"


# =========================
# MAIN
# =========================
def main():

    print("📥 Loading datasets...")
    df = get_combined_dataset(30000)

    print("⚖️ Balancing dataset...")
    df = balance_dataset(df)

    print(df["Label"].value_counts())

    print("🔍 Building retriever...")
    vectorizer, X, texts, labels = build_retriever(df)

    print("📥 Loading test set...")

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "data" / "subm2.csv"
    print(model_path)


    test_df = pd.read_csv(model_path, sep=";")

    predictions = []

    print("🚀 Running RAG + Ollama...")

    for i, row in test_df.iterrows():
        text = row["Text"]

        examples = retrieve_examples(text, vectorizer, X, texts, labels, k=3)
        prompt = build_prompt(text, examples)

        try:
            raw = classify_with_ollama(prompt, model="mistral")
            label = clean_label(raw)
        except Exception as e:
            print("Erro:", e)
            label = "Human"

        predictions.append(label)

        if i % 10 == 0:
            print(f"{i}/{len(test_df)}")

    if "Id" in test_df.columns:
        id_col = test_df["Id"]
    elif "id" in test_df.columns:
        id_col = test_df["id"]
    else:
        id_col = pd.Series(range(len(test_df)))

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "Text": test_df["Text"],
        "Label": predictions
    })

    model_path_write = project_root / "data" / "subm2_pred.csv"
    submission.to_csv(model_path_write, index=False,quoting=csv.QUOTE_NONE, sep=";", escapechar="\\")
    print("subm2_pred.csv criado!")

if __name__ == "__main__":
    main()