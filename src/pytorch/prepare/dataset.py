import torch
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(self,X,y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze()
        y = self.y[idx]

        return x, y
    

def build_vectorizer():

    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1,2),
        min_df=5,
        stop_words="english"
    )


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
        # "mistralai": "Mistral",
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
    # df = pd.concat([df_otb, df_atdp, df_ap], ignore_index=True)

    return df