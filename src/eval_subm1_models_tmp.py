import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models.dnn.model import NumpyModel
from models.pytorch.model import PyTorchModel
from models.transformer.model import TransformModel
from models.bert.model import BertModel


def main() -> None:
    df = pd.read_csv("data/subm1_labels_revealed.csv", sep=";")
    texts = df["Text"].astype(str).tolist()
    labels = df["Label"].astype(str).tolist()

    models = [
        ("numpy-dnn", "../models/numpy-dnn.pkl.gz", NumpyModel.load),
        ("pytorch-dnn", "../models/pytorch-dnn.pt", PyTorchModel.load),
        ("transformer", "../models/transformer.pt", TransformModel.load),
        ("bert", "../models/bert.pt", BertModel.load),
    ]

    for name, path, loader in models:
        try:
            model = loader(path)
            preds = model.predict(texts)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            print(f"{name}: acc={acc:.4f} f1_macro={f1:.4f}")
        except Exception as exc:
            print(f"{name}: ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
