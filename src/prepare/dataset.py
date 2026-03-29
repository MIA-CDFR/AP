import pandas as pd

from datasets import load_dataset
from pathlib import Path


TARGET_LABELS = ["Anthropic", "Google", "Human", "Meta", "OpenAI"]


def _sample_n(df: pd.DataFrame, n_lines: int, random_state: int = 42) -> pd.DataFrame:
    n_lines = min(n_lines, len(df))
    return df.sample(n_lines, random_state=random_state).reset_index(drop=True)


def get_prof_dataset(n_lines: int = 125) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset-exemplos.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df["id"] = df["ID"]
    df = df[df["Text"].notna() & df["Label"].notna()]
    return _sample_n(df[["id", "Text", "Label"]], n_lines=n_lines)


def get_subm1_dataset(n_lines: int = 100) -> pd.DataFrame:
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
    df["id"] = df.index.astype(str)
    return _sample_n(df[["id", "Text", "Label"]].dropna(), n_lines=n_lines)


def get_subm2_dataset(n_lines: int = 100) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "subm2_labels_revealed.csv"

    df = pd.read_csv(dataset_path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    required = {"Text", "Label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns {required}. Found columns: {list(df.columns)}"
        )
    df = df[df["Text"].notna() & df["Label"].notna()]
    df["id"] = df.index.astype(str)
    return _sample_n(df[["id", "Text", "Label"]].dropna(), n_lines=n_lines)


def get_otb_dataset(n_lines: int = 74566) -> pd.DataFrame:
    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    df = pd.concat([df_train, df_test], ignore_index=True)

    mapping_classes = {
        "meta-llama": "Meta",
        "google": "Google",
    }

    model_family = df["model"].str.split("/").str[0].str.lower()
    df = df[model_family.isin(mapping_classes.keys())].copy()
    df["id"] = df["url"]
    df["Text"] = df["content"]
    df["Label"] = model_family[model_family.isin(mapping_classes.keys())].map(mapping_classes)
    df = df[df["Text"].notna() & df["Label"].notna()]
    return _sample_n(df[["id", "Text", "Label"]], n_lines=n_lines)


def get_openai_dataset(n_lines: int = 33203) -> pd.DataFrame:
    dataset = load_dataset("Dahoas/instruct-synthetic-prompt-responses")

    df = dataset["train"].to_pandas()

    df["id"] = df.index.astype(str)
    df["Text"] = df["response"]
    df["Label"] = df["response"].apply(lambda x: "OpenAI" if x.strip() != "" else "Others")
    df = df[df["Label"] != "Others"]
    df = df[df["Text"].notna() & df["Label"].notna()]
    return _sample_n(df[["id", "Text", "Label"]], n_lines=n_lines)


def get_atdp_dataset(n_lines: int = 37283) -> pd.DataFrame:
    dataset = load_dataset("artem9k/ai-text-detection-pile")

    # Select only the rows we need before converting to pandas.
    # The first rows in this dataset are human-authored; loading the full
    # 1.4M-row table to pandas (~8 GB) before filtering would cause OOM.
    n_take = min(n_lines, len(dataset["train"]))
    df = dataset["train"].select(range(n_take)).to_pandas()

    df["Text"] = df["text"]
    df = df[df["source"] == "human"]
    df["Label"] = df["source"].apply(lambda x: "Human" if x == "human" else "")
    df = df[df["Text"].notna() & df["Label"].notna()]
    return _sample_n(df[["id", "Text", "Label"]], n_lines=n_lines)


def get_ap_dataset(n_lines: int = 3360) -> pd.DataFrame:
    dataset = load_dataset("Anthropic/persuasion")

    df = dataset["train"].to_pandas()

    df = df[df["source"].str.startswith("Claude")].copy()
    df["id"] = df["worker_id"]
    df["Text"] = df["argument"]
    df["Label"] = "Anthropic"
    df = df[df["Text"].notna() & df["Label"].notna()]
    return _sample_n(df[["id", "Text", "Label"]], n_lines=n_lines)


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
    return _sample_n(df[["id", "Text", "Label"]], n_lines=n_lines)


def _balance_by_label(
    df: pd.DataFrame,
    target_per_class: int = 37283,
    labels: list[str] = TARGET_LABELS,
    random_state: int = 42,
) -> pd.DataFrame:
    balanced_parts = []
    for label in labels:
        subset = df[df["Label"] == label]
        if subset.empty:
            continue
        replace = len(subset) < target_per_class
        sampled = subset.sample(target_per_class, replace=replace, random_state=random_state)
        balanced_parts.append(sampled)

    if not balanced_parts:
        return df.reset_index(drop=True)

    return pd.concat(balanced_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def get_submission_validation_dataset(submission_round: int = 1, random_state: int = 42) -> pd.DataFrame:
    if submission_round == 1:
        return get_prof_dataset(n_lines=125)
    if submission_round == 2:
        return get_subm1_dataset(n_lines=100).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if submission_round == 3:
        return get_subm2_dataset(n_lines=100).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    raise ValueError("submission_round must be 1, 2, or 3")


def get_datasets(
    include_subm1: bool = False,
    include_subm2: bool = False,
    submission_round: int | None = None,
    balance: bool = True,
    target_per_class: int = 37283,
    include_agnews: bool = False,
) -> pd.DataFrame:
    if submission_round is not None:
        include_subm1 = submission_round >= 2
        include_subm2 = submission_round >= 3

    df_prof = get_prof_dataset()
    df_otb = get_otb_dataset()
    df_atdp = get_atdp_dataset(n_lines=target_per_class * 2)
    df_ap = get_ap_dataset()
    df_openai = get_openai_dataset()
    parts = [df_prof, df_otb, df_atdp, df_ap, df_openai]

    if include_subm1:
        parts.append(get_subm1_dataset())

    if include_subm2:
        parts.append(get_subm2_dataset())

    if include_agnews:
        parts.append(get_agnews_dataset())

    df = pd.concat(parts, ignore_index=True)
    df = df[df["Label"].isin(TARGET_LABELS)].reset_index(drop=True)

    if balance:
        df = _balance_by_label(df, target_per_class=target_per_class, labels=TARGET_LABELS)

    return df
