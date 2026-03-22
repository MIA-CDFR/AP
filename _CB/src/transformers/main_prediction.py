# Este main_prediction.py já contem o código para o BERT tb
# O código para o transformer de raíz sem o BERT está em main_prediction_Copy.py

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.transformers import TransformerClassifier
from prepare.feature import (
    build_handcrafted_matrix,
    preprocess_text,
    preprocess_text_clean,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    checkpoint_path = module_path / "trained_models" / "bert_classifier.pth"
    if not checkpoint_path.exists():
        checkpoint_path = module_path / "trained_models" / "transformer.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_map = checkpoint["label_map"]
    idx_to_label = {v: k for k, v in label_map.items()}
    model_type = checkpoint.get("model_type", "transformer")

    if model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint["model_name"],
            num_labels=len(label_map),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    else:
        input_dim = checkpoint["input_dim"]
        n_classes = len(label_map)
        seq_len_model = checkpoint["seq_len"]

        model = TransformerClassifier(input_dim, n_classes, seq_len=seq_len_model).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        vectorizer = checkpoint["vectorizer"]

    import pandas as pd

    df_new = pd.read_csv(f"{module_path}/data/subm2.csv", sep=";")
    texts = df_new["Text"].tolist()

    if model_type == "bert":
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=checkpoint["max_length"],
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            ).logits
            preds = logits.argmax(dim=1).cpu().numpy()
    else:
        texts_clean = [preprocess_text(t) for t in texts]
        clean_light = [preprocess_text_clean(t) for t in texts]

        X_tfidf = vectorizer.transform(texts_clean)
        X_hand, _ = build_handcrafted_matrix(texts, clean_light)

        mean = checkpoint["hand_mean"]
        std = checkpoint["hand_std"]
        X_hand = (X_hand - mean) / std

        X = np.hstack([X_tfidf.toarray(), X_hand])

        seq_len = checkpoint["seq_len"]
        pad_size = (seq_len - (X.shape[1] % seq_len)) % seq_len
        if pad_size > 0:
            X = np.hstack([X, np.zeros((X.shape[0], pad_size))])

        embed_dim = X.shape[1] // seq_len
        X = X.reshape(-1, seq_len, embed_dim)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()

    pred_labels = [idx_to_label[p] for p in preds]
    df_new["Prediction"] = pred_labels
    df_new.to_csv(f"{module_path}/data/predictions2.csv", index=False)

    print("Predictions saved!")
    print("Evaluation/Prediction complete.")


if __name__ == "__main__":
    main()
