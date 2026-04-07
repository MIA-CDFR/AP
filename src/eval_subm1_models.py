import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.dnn.model import NumpyModel
from models.pytorch.model import PyTorchModel
from models.transformer.model import TransformModel
from models.bert.model import BertModel


DISPLAY_NAMES = {
    "baseline": "Baseline Linear",
    "numpy-dnn": "NumPy DNN",
    "pytorch-dnn": "PyTorch DNN",
    "transformer": "Transformer Genérico",
    "bert": "RoBERTa-base (fine-tuning)",
}


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

    results = [
        {
            "model_id": "baseline",
            "model": DISPLAY_NAMES["baseline"],
            "accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "status": "not available",
        }
    ]

    for name, path, loader in models:
        try:
            model = loader(path)
            preds = model.predict(texts)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            precision = precision_score(labels, preds, average="macro", zero_division=0)
            recall = recall_score(labels, preds, average="macro", zero_division=0)
            results.append(
                {
                    "model_id": name,
                    "model": DISPLAY_NAMES[name],
                    "accuracy": acc,
                    "macro_f1": f1,
                    "macro_precision": precision,
                    "macro_recall": recall,
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_id": name,
                    "model": DISPLAY_NAMES[name],
                    "accuracy": None,
                    "macro_f1": None,
                    "macro_precision": None,
                    "macro_recall": None,
                    "status": f"ERROR {type(exc).__name__}: {exc}",
                }
            )

    results_df = pd.DataFrame(results)

    print("\n=== Metrics Table ===")
    printable_df = results_df.copy()
    for column in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        printable_df[column] = printable_df[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "--")
    print(printable_df[["model", "accuracy", "macro_f1", "macro_precision", "macro_recall", "status"]].to_string(index=False))

    print("\n=== LaTeX Rows ===")
    for row in results:
        if row["status"] == "ok":
            print(
                f'{row["model"]} & '
                f'{row["accuracy"]:.4f} & '
                f'{row["macro_f1"]:.4f} & '
                f'{row["macro_precision"]:.4f} & '
                f'{row["macro_recall"]:.4f} \\\\'
            )
        else:
            print(f'{row["model"]} & -- & -- & -- & -- \\\\ % {row["status"]}')

    output_dir = Path("../models/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subm1_eval_metrics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nCSV saved to: {output_path}")


if __name__ == "__main__":
    main()
