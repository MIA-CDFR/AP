
import numpy as np


import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pytorch.models.dnn import DNNClassifier
from pytorch.prepare.model import evaluate, train_model
from pytorch.prepare.dataset import TextDataset, get_datasets
from pytorch.prepare.feature import (
    preprocess_text, preprocess_text_clean,
    build_vectorizer, encode_labels,
    build_handcrafted_matrix, standardize_train_test,
)


def main():
    df = get_datasets()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Keep raw texts for handcrafted features before cleaning
    raw_train = df_train["Text"].tolist()
    raw_test = df_test["Text"].tolist()

    # TF-IDF uses aggressive cleaning
    df_train["Text_tfidf"] = df_train["Text"].apply(preprocess_text)
    df_test["Text_tfidf"] = df_test["Text"].apply(preprocess_text)

    # Handcrafted features use light cleaning (preserves structure)
    clean_train = [preprocess_text_clean(t) for t in raw_train]
    clean_test = [preprocess_text_clean(t) for t in raw_test]

    # --- TF-IDF features ---
    vectorizer = build_vectorizer()
    X_tfidf_train = vectorizer.fit_transform(df_train["Text_tfidf"])
    X_tfidf_test = vectorizer.transform(df_test["Text_tfidf"])

    # --- Handcrafted features ---
    X_hand_train, feature_names = build_handcrafted_matrix(raw_train, clean_train)
    X_hand_test, _ = build_handcrafted_matrix(raw_test, clean_test)
    X_hand_train, X_hand_test, hand_mean, hand_std = standardize_train_test(X_hand_train, X_hand_test)

    # --- Combine: TF-IDF (dense) + handcrafted ---
    X_train = np.hstack([X_tfidf_train.toarray(), X_hand_train])
    X_test = np.hstack([X_tfidf_test.toarray(), X_hand_test])

    print(f"TF-IDF features:      {X_tfidf_train.shape[1]}")
    print(f"Handcrafted features: {X_hand_train.shape[1]} {feature_names}")
    print(f"Total features:       {X_train.shape[1]}")

    y_train, label_map = encode_labels(df_train["Label"])
    y_test = np.array([label_map[l] for l in df_test["Label"]])

    n_classes = len(label_map)

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DNNClassifier(X_train.shape[1], n_classes=n_classes).to(device)

    model = train_model(model, train_loader, test_loader, device)

    from pathlib import Path

    path = Path(__file__).resolve().parent

    torch.save({
        "model_type": "dnn",
        "model_state": model.state_dict(),
        "label_map": label_map,
        "vectorizer": vectorizer,
        "input_dim": X_train.shape[1],
        "n_hand_features": X_hand_train.shape[1],
        "hand_feature_names": feature_names,
        "hand_mean": hand_mean,
        "hand_std": hand_std,
    }, path / "model.pth")

    print("Model saved.")

    acc, y_true, y_pred = evaluate(model, test_loader, device)

    print("\nFinal accuracy:", acc)

    cm = confusion_matrix(y_true, y_pred)

    print(cm)

if __name__ == "__main__":
    main()
