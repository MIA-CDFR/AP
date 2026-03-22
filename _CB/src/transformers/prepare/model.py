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
from pytorch.prepare.feature import (
    preprocess_text,
    preprocess_text_clean,
    build_handcrafted_matrix,
)


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
    model_type = checkpoint.get("model_type", "dnn")
    n_hand_features = checkpoint.get("n_hand_features", 0)
    hand_feature_names = checkpoint.get("hand_feature_names", [])
    hand_mean = checkpoint.get("hand_mean", None)
    hand_std = checkpoint.get("hand_std", None)
    print(f"Loaded model type from checkpoint: {model_type}")

    tfidf_dim = len(vectorizer.get_feature_names_out())
    expected_input_dim = tfidf_dim + n_hand_features

    if "input_dim" in checkpoint:
        input_dim = int(checkpoint["input_dim"])
    else:
        input_dim = expected_input_dim
        if "model_state" in checkpoint:
            first_2d_weight = next(
                (v for v in checkpoint["model_state"].values() if getattr(v, "ndim", 0) == 2),
                None,
            )
            if first_2d_weight is not None:
                state_input_dim = int(first_2d_weight.shape[1])
                if state_input_dim != input_dim:
                    input_dim = state_input_dim

    n_classes = len(label_map)

    model = build_model_from_type(model_type, input_dim, n_classes)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    model.hand_mean = hand_mean
    model.hand_std = hand_std
    model.n_hand_features = n_hand_features
    model.hand_feature_names = hand_feature_names

    return model, vectorizer, label_map, model_type, n_hand_features, hand_feature_names


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


def evaluate_dataset(
    model,
    vectorizer,
    label_map,
    csv_path,
    output_path=None,
    n_hand_features=0,
    hand_feature_names=None,
    hand_mean=None,
    hand_std=None,
):

    print(f"\nLoading dataset: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    print(f"Dataset loaded: {len(df)} samples")

    df["Text_Clean"] = df["Text"].apply(preprocess_text)
    X_tfidf = vectorizer.transform(df["Text_Clean"]).toarray()

    hand_feature_names = hand_feature_names or []

    if (n_hand_features is None or n_hand_features == 0) and hasattr(model, "n_hand_features"):
        n_hand_features = int(getattr(model, "n_hand_features", 0) or 0)
    if not hand_feature_names and hasattr(model, "hand_feature_names"):
        hand_feature_names = list(getattr(model, "hand_feature_names", []) or [])

    tfidf_dim = X_tfidf.shape[1]
    first_2d_weight = next((p for p in model.parameters() if getattr(p, "ndim", 0) == 2), None)
    model_input_dim = int(first_2d_weight.shape[1]) if first_2d_weight is not None else tfidf_dim

    inferred_n_hand = max(0, model_input_dim - tfidf_dim)
    if (n_hand_features is None or n_hand_features == 0) and inferred_n_hand > 0:
        n_hand_features = inferred_n_hand

    use_handcrafted = (n_hand_features is not None and n_hand_features > 0)

    if hand_mean is None and hasattr(model, "hand_mean"):
        hand_mean = model.hand_mean
    if hand_std is None and hasattr(model, "hand_std"):
        hand_std = model.hand_std

    if use_handcrafted:
        raw_texts = df["Text"].astype(str).tolist()
        clean_texts = [preprocess_text_clean(t) for t in raw_texts]

        X_hand, inferred_feature_names = build_handcrafted_matrix(raw_texts, clean_texts)

        if hand_feature_names:
            if set(hand_feature_names) != set(inferred_feature_names):
                missing = set(hand_feature_names) - set(inferred_feature_names)
                extra = set(inferred_feature_names) - set(hand_feature_names)
                raise ValueError(
                    "Handcrafted feature mismatch between checkpoint and runtime. "
                    f"Missing: {sorted(missing)} | Extra: {sorted(extra)}"
                )
            name_to_idx = {name: i for i, name in enumerate(inferred_feature_names)}
            X_hand = X_hand[:, [name_to_idx[name] for name in hand_feature_names]]

        if hand_mean is not None and hand_std is not None:
            hand_mean = np.asarray(hand_mean)
            hand_std = np.asarray(hand_std)
            hand_std[hand_std == 0] = 1
            X_hand = (X_hand - hand_mean) / hand_std
        else:
            mean = X_hand.mean(axis=0, keepdims=True)
            std = X_hand.std(axis=0, keepdims=True)
            std[std == 0] = 1
            X_hand = (X_hand - mean) / std

        X_np = np.hstack([X_tfidf, X_hand])
    else:
        X_np = X_tfidf

    X = torch.tensor(X_np, dtype=torch.float32)
    
    if "Label" in df.columns:
        y_true = np.array([label_map[l] for l in df["Label"]])
    else:
        y_true = None

    with torch.no_grad():

        outputs = model(X)

        preds = torch.argmax(outputs, dim=1).numpy()

    accuracy = None

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