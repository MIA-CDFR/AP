import pandas as pd

from datasets import load_dataset
from pathlib import Path


def get_prof_dataset(n_lines: int = 10000) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset-exemplos.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df["id"] = df["ID"]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_subm1_dataset(n_lines: int = 10000) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "subm1_labels_revealed.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    required = {"Text", "Label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns {required}. Found columns: {list(df.columns)}"
        )

    return df[["Text", "Label"]].dropna().sample(n_lines, random_state=42).reset_index(drop=True)


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
