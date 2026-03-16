
import numpy as np


import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pytorch.models.dnn import DNNClassifier
from pytorch.prepare.model import evaluate, train_model
from pytorch.prepare.dataset import TextDataset, get_datasets
from pytorch.prepare.feature import preprocess_text, build_vectorizer, encode_labels


def main():
    df = get_datasets()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train["Text"] = df_train["Text"].apply(preprocess_text)
    df_test["Text"] = df_test["Text"].apply(preprocess_text)

    vectorizer = build_vectorizer()

    X_train = vectorizer.fit_transform(df_train["Text"])
    X_test = vectorizer.transform(df_test["Text"])


    y_train, label_map = encode_labels(df_train["Label"])
    y_test = np.array([label_map[l] for l in df_test["Label"]])

    n_classes = len(label_map)

    train_dataset = TextDataset(X_train,y_train)
    test_dataset = TextDataset(X_test,y_test)

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DNNClassifier(X_train.shape[1],n_classes=n_classes).to(device)

    model = train_model(model, train_loader, test_loader, device)

    from pathlib import Path

    path = Path(__file__).resolve().parent

    torch.save({
        "model_type": "dnn",
        "model_state": model.state_dict(),
        "label_map": label_map,
        "vectorizer": vectorizer
    }, path / "model.pth")

    print("Model saved.")

    acc, y_true, y_pred = evaluate(model,test_loader,device)

    print("\nFinal accuracy:",acc)

    cm = confusion_matrix(y_true, y_pred)

    print(cm)

if __name__ == "__main__":
    main()
