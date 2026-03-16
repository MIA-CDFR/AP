import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

from pytorch.models.dnn import DNNClassifier
from pytorch.models.lstm import LSTMClassifier
from pytorch.models.logistic import LogisticRegression
from pytorch.models.linear import LinearClassifier
from pytorch.models.gru import GRUClassifier
from pytorch.prepare.feature import preprocess_text, build_vectorizer, encode_labels


class EarlyStopping:
    def __init__(self,patience=3):
        self.patience = patience
        self.best = None
        self.counter = 0

    def step(self,metric):
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
            return False

        else:
            self.counter += 1
            return self.counter >= self.patience


def train_epoch(model,loader,criterion,optimizer,device):
    model.train()
    total_loss = 0

    for X,y in loader:
        X,y = X.to(device),y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(loader)


def evaluate(model,loader,device):

    model.eval()

    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device),y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs,dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return correct/total,np.array(y_true),np.array(y_pred)


def train_model(model,train_loader,test_loader,device,epochs=20):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter()

    early_stop = EarlyStopping()

    for epoch in range(epochs):

        loss = train_epoch(model,train_loader,criterion,optimizer,device)

        acc,_,_ = evaluate(model,test_loader,device)

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)

        print(f"Epoch {epoch+1} | loss {loss:.4f} | acc {acc:.4f}")

        if early_stop.step(acc):

            print("Early stopping triggered")
            break

    return model

def build_model_from_type(model_type, input_dim, n_classes):

    model_builders = {
        "gru": lambda: GRUClassifier(input_dim, n_classes=n_classes),
        "linear": lambda: LinearClassifier(input_dim, n_classes=n_classes),
        "dnn": lambda: DNNClassifier(input_dim, n_classes=n_classes),
        "lstm": lambda: LSTMClassifier(input_dim, n_classes=n_classes),
        "logistic_regression": lambda: LogisticRegression(input_dim, n_classes=n_classes),
    }

    normalized_model_type = str(model_type).strip().lower()

    if normalized_model_type not in model_builders:
        available = ", ".join(sorted(model_builders))
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: {available}")

    return model_builders[normalized_model_type]()


def load_model(model_path):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    vectorizer = checkpoint["vectorizer"]
    label_map = checkpoint["label_map"]
    model_type = checkpoint.get("model_type")
    print(f"Loaded model type from checkpoint: {model_type}")
    
    input_dim = len(vectorizer.get_feature_names_out())
    n_classes = len(label_map)

    model = build_model_from_type(model_type, input_dim, n_classes)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    return model, vectorizer, label_map, model_type


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


def evaluate_dataset(model, vectorizer, label_map, csv_path, output_path=None):

    print(f"\nLoading dataset: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    print(f"Dataset loaded: {len(df)} samples")

    df["Text_Clean"] = df["Text"].apply(preprocess_text)

    X = vectorizer.transform(df["Text_Clean"])

    X = torch.tensor(X.toarray(), dtype=torch.float32)
    
    if "Label" in df.columns:
        y_true = np.array([label_map[l] for l in df["Label"]])
    else:
        y_true = None

    with torch.no_grad():

        outputs = model(X)

        preds = torch.argmax(outputs, dim=1).numpy()

    accuracy = np.mean(preds == y_true)

    if y_true is not None:
        accuracy = np.mean(preds == y_true)
        print("Accuracy:", accuracy)

        cm = confusion_matrix(y_true, preds)

        print(cm)

        plot_confusion_matrix(cm, list(label_map.keys()))        


    inv_labels = {v:k for k,v in label_map.items()}
    df["Labels"] = [inv_labels[p] for p in preds]

    if output_path:
        df[["ID", "Text", "Labels"]].to_csv(output_path, sep=";", index=False)
        print("Predictions saved to submission.csv")

    return accuracy