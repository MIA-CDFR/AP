from pathlib import Path

import torch
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# =====================
# CONFIG
# =====================
MODEL_PATH =  module_path / "trained_models" / "best_model.pth"
CHECKPOINT = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# TOKENIZE
# =====================
def bert_tokenize(texts, tokenizer, max_length=MAX_LEN):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


# =====================
# LOAD MODEL
# =====================
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Run test-bert.py first to generate it."
        )

    checkpoint_data = torch.load(MODEL_PATH, map_location=device)

    label_map = checkpoint_data["label_map"]
    checkpoint_name = checkpoint_data["checkpoint"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_name,
        num_labels=len(label_map)
    )

    model.load_state_dict(checkpoint_data["model_state"])
    model.to(device)
    model.eval()

    return model, tokenizer, label_map


# =====================
# PREDICT CSV
# =====================
def predict_csv(input_path, output_path):
    model, tokenizer, label_map = load_model()

    df = pd.read_csv(input_path, sep=";")
    texts = df["Text"].astype(str).tolist()

    encodings = bert_tokenize(texts, tokenizer)

    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    preds = []

    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    # converter índices → labels
    idx_to_label = {v: k for k, v in label_map.items()}
    df["Prediction"] = [idx_to_label[p] for p in preds]

    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# =====================
# MAIN
# =====================
def main():
    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    input_file = f"{module_path}/data/subm2.csv"   # muda se quiseres
    output_file = f"{module_path}/data/predictions2.csv"

    predict_csv(input_file, output_file)


if __name__ == "__main__":
    main()
