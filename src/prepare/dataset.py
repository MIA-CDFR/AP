import pandas as pd

from datasets import load_dataset
from pathlib import Path


def get_prof_dataset(n_lines: int = 10000) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset-exemplos.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df["id"] = df["ID"]
    df = df[df["Text"].notna() & df["Label"].notna()]
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
    df = df[df["Text"].notna() & df["Label"].notna()]
    n_lines = min(n_lines, len(df))

    return df[["Text", "Label"]].dropna().sample(n_lines, random_state=42).reset_index(drop=True)


def get_otb_dataset(n_lines: int = 10000) -> pd.DataFrame:
    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    df = pd.concat([df_train, df_test], ignore_index=True)

    mapping_classes = {
        "meta-llama": "Meta",
        # "qwen": "OpenAI",
        "google": "Google",
        "anthropic": "Anthropic",
    }

    df["id"] = df["url"]
    df["Text"] = df["content"]
    df["Label"] = df["model"].apply(lambda x: mapping_classes.get(x.split("/")[0].lower(), "Others"))
    df = df[df["Label"] != "Others"]
    df = df[df["Text"].notna() & df["Label"].notna()]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_openai_dataset(n_lines: int = 5000) -> pd.DataFrame:
    dataset = load_dataset("Dahoas/instruct-synthetic-prompt-responses")

    df = dataset["train"].to_pandas()

    df["id"] = df.index.astype(str)
    df["Text"] = df["response"]
    df["Label"] = df["response"].apply(lambda x: "OpenAI" if x.strip() != "" else "Others")
    df = df[df["Label"] != "Others"]
    df = df[df["Text"].notna() & df["Label"].notna()]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)

def get_atdp_dataset(n_lines: int = 5000) -> pd.DataFrame:
    dataset = load_dataset("artem9k/ai-text-detection-pile")

    df = dataset["train"].to_pandas()

    df["Text"] = df["text"]
    df = df[df["source"] == "human"]
    df["Label"] = df["source"].apply(lambda x: "Human" if x == "human" else "")
    df = df[df["Text"].notna() & df["Label"].notna()]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_ap_dataset(n_lines: int = 5000) -> pd.DataFrame:
    dataset = load_dataset("Anthropic/persuasion")

    df = dataset["train"].to_pandas()

    df["id"] = df["worker_id"]
    df["Text"] = df["argument"]
    df["Label"] = df["source"].apply(lambda x: "Anthropic" if x.startswith("Claude") else "Human")
    df = df[df["Text"].notna() & df["Label"].notna()]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_agnews_dataset(n_lines: int = 10000) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "ag_news_rephrased.csv"

    df = pd.read_csv(dataset_path)
    df_openai = df.copy()
    df_openai["id"] = df_openai.index.astype(str)
    df_openai["Text"] = df_openai["description_rephrased_openai"]
    df_openai["Label"] = "OpenAI"

    df_meta = df.copy()
    df_meta["id"] = df_meta.index.astype(str)
    df_meta["Text"] = df_meta["description_rephrased_meta"]
    df_meta["Label"] = "Meta"

    df_google = df.copy()
    df_google["id"] = df_google.index.astype(str)
    df_google["Text"] = df_google["description_rephrased_google"]
    df_google["Label"] = "Google"

    df = pd.concat([df_openai, df_meta, df_google], ignore_index=True)
    df = df[df["Text"].notna() & df["Label"].notna()]
    n_lines = min(n_lines, len(df))

    return df[["id", "Text", "Label"]].sample(n_lines, random_state=42).reset_index(drop=True)


def get_datasets(include_subm1: bool = False) -> pd.DataFrame:
    df_prof = get_prof_dataset()
    df_otb = get_otb_dataset()
    df_atdp = get_atdp_dataset()
    df_ap = get_ap_dataset()
    df_openai = get_openai_dataset()
    df_agnews = get_agnews_dataset()

    df_subm1 = get_subm1_dataset() if include_subm1 else pd.DataFrame(columns=["id", "Text", "Label"])

    df = pd.concat([df_prof, df_otb, df_atdp, df_ap, df_openai, df_subm1, df_agnews], ignore_index=True)

    return df
