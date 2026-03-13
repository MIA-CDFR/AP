import torch
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch.nn as nn

class GRUClassifier(nn.Module):

    def __init__(self, input_dim, embed_dim=128, hidden_dim=128, n_classes=6):

        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        output, h = self.gru(x)

        return self.fc(h[-1])

############################################
# TEXT PREPROCESSING (igual ao treino)
############################################

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


############################################
# CONFUSION MATRIX
############################################

def confusion_matrix(y_true, y_pred, n_classes):

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return cm


def plot_confusion_matrix(cm, labels):

    fig, ax = plt.subplots(figsize=(8,6))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


############################################
# LOAD MODEL
############################################

import torch

def load_model(model_path):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    vectorizer = checkpoint["vectorizer"]
    label_map = checkpoint["label_map"]
    
    input_dim = len(vectorizer.get_feature_names_out())
    n_classes = len(label_map)

    model = GRUClassifier(input_dim, n_classes=n_classes)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    return model, vectorizer, label_map


############################################
# EVALUATE DATASET
############################################

def evaluate_dataset(model, vectorizer, label_map, csv_path):

    print(f"\nLoading dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples")

    df["Text"] = df["Text"].apply(preprocess_text)

    X = vectorizer.transform(df["Text"])

    X = torch.tensor(X.toarray(), dtype=torch.float32)
    
    y_true = np.array([label_map[l] for l in df["Label"]])

    with torch.no_grad():

        outputs = model(X)

        preds = torch.argmax(outputs, dim=1).numpy()

    accuracy = np.mean(preds == y_true)

    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_true, preds, len(label_map))

    plot_confusion_matrix(cm, list(label_map.keys()))

    return accuracy


############################################
# MAIN
############################################

def main():

    model_path = "./models/model.pth"

    dataset_path = "./models/dataset-exemplos.csv"

    model, vectorizer, label_map = load_model(model_path)

    evaluate_dataset(
        model,
        vectorizer,
        label_map,
        dataset_path
    )


if __name__ == "__main__":
    main()