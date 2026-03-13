from pathlib import Path

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()

module_path = Path(__file__).resolve().parent
dataset_path = module_path / "data" / "dataset-hf.csv"

df = pd.concat([df_train, df_test], ignore_index=True)

mapping_classes = {
    "meta-llama": "Meta",
    "qwen": "OpenAI",
    "mistralai": "Mistral",
    "google": "Google",
    "anthropic": "Anthropic",
}

df["Text"] = df["content"]
df["Label"] = df["model"].apply(lambda x: mapping_classes.get(x.split("/")[0].lower(), "Others"))

df[["url", "Text", "Label"]].head(1000).to_csv(dataset_path, index=False)
