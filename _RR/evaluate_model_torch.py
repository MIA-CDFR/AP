import torch
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
import csv

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

class LinearClassifier(nn.Module):

    def __init__(self,input_dim,n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim,n_classes)

    def forward(self,x):
        return self.fc(x)

class LogisticRegression(nn.Module):

    def __init__(self,input_dim,n_classes):

        super().__init__()

        self.linear = nn.Linear(input_dim,n_classes)

    def forward(self,x):

        return self.linear(x)


class DNNClassifier(nn.Module):

    def __init__(self,input_dim,n_classes):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,n_classes)

        )

    def forward(self,x):

        return self.net(x)
    
class LSTMClassifier(nn.Module):

    def __init__(self,input_dim,embed_dim=128,hidden_dim=128,n_classes=6):

        super().__init__()

        self.embedding = nn.Linear(input_dim,embed_dim)

        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)

        self.fc = nn.Linear(hidden_dim,n_classes)

    def forward(self,x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        output,(h,c) = self.lstm(x)

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

    #model = GRUClassifier(input_dim, n_classes=n_classes)
    #model = LinearClassifier(input_dim, n_classes=n_classes)
    #model = DNNClassifier(input_dim, n_classes=n_classes)
    #model = LSTMClassifier(input_dim, n_classes=n_classes)
    model = LogisticRegression(input_dim, n_classes=n_classes)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    return model, vectorizer, label_map


############################################################
# CSV VALIDATION
############################################################

def read_csv_smart(csv_path):

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    delimiter = None
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        pass

    if delimiter:
        df = pd.read_csv(csv_path, sep=delimiter, encoding="utf-8-sig")
    else:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    # Fallback for files that were parsed into a single "ID,Text,Label" column.
    if len(df.columns) == 1 and "," in df.columns[0]:
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
        df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    return df

############################################
# EVALUATE DATASET
############################################

def evaluate_dataset(model, vectorizer, label_map, csv_path, output_path=None):

    print(f"\nLoading dataset: {csv_path}")

    #df = pd.read_csv(csv_path)
    df = read_csv_smart(csv_path)

    print(f"Dataset loaded: {len(df)} samples")

    df["Text"] = df["Text"].apply(preprocess_text)

    X = vectorizer.transform(df["Text"])

    X = torch.tensor(X.toarray(), dtype=torch.float32)
    
    #y_true = np.array([label_map[l] for l in df["Label"]])
    if "Label" in df.columns:
        y_true = np.array([label_map[l] for l in df["Label"]])
    else:
        y_true = None

    with torch.no_grad():

        outputs = model(X)

        preds = torch.argmax(outputs, dim=1).numpy()

    accuracy = np.mean(preds == y_true)

    #print("Accuracy:", accuracy)
    # if y_true is not None:
    #     accuracy = np.mean(preds == y_true)
    #     print("Accuracy:", accuracy)
    #     cm = (y_true, preds, len(label_map))
    #     print(cm)
    #     plot_confusion_matrix(cm, list(label_map.keys()))

    if y_true is not None:
        accuracy = np.mean(preds == y_true)
        print("Accuracy:", accuracy)

        cm = confusion_matrix(y_true, preds, len(label_map))

        print(cm)

        plot_confusion_matrix(cm, list(label_map.keys()))        
    #cm = confusion_matrix(y_true, preds, len(label_map))

    #plot_confusion_matrix(cm, list(label_map.keys()))
    #confusion_matrix(y_true, preds, len(label_map))

    inv_labels = {v:k for k,v in label_map.items()}
    df["Prediction"] = [inv_labels[p] for p in preds]

    if output_path:
        df.to_csv(output_path, index=False)
        print("Predictions saved to submission.csv")

    return accuracy


############################################
# MAIN
############################################

def main():

    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    dataset_path_model = module_path / "models"
    Path(dataset_path_model).mkdir(parents=True, exist_ok=True)
    dataset_path_validate = module_path / "data"

    model_path = f"{dataset_path_model}\model.pth"

    print(f"Model path: {model_path}")
    dataset_path = f"{dataset_path_validate}\dataset-exemplos.csv"
    print(f"Dataset path: {dataset_path}")
    model, vectorizer, label_map = load_model(model_path)

    evaluate_dataset(
        model,
        vectorizer,
        label_map,
        dataset_path,
        None
    )


    dataset_path2 = f"{dataset_path_validate}\subm1.csv"
    dataset_path3 = f"{dataset_path_validate}\subm1_result.csv"
    evaluate_dataset(
        model,
        vectorizer,
        label_map,
        dataset_path2,
        output_path=dataset_path3
    )

    evaluate_dataset(
        model,
        vectorizer,
        label_map,
        dataset_path3,
        None
    )

if __name__ == "__main__":
    main()