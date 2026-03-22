import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare.dataset import get_datasets
from prepare.feature import encode_labels


# Q: Para usar o modelo distilbert-base-uncased, basta alterar o model_name?
# R: Quase, mas no teu script atual sim, praticamente basta mudar:
# _CB/src/transformers/main_train_BERT.py
# de:   MODEL_NAME = "bert-base-uncased"
# para: MODEL_NAME = "distilbert-base-uncased"
#
# Como estás a usar AutoTokenizer e AutoModelForSequenceClassification, o resto adapta-se automaticamente.
# Só há 3 notas:
# Apaga ou muda o nome do checkpoint final, para não confundir modelos diferentes.
# Se já tinhas um bert_classifier.pth, o main_prediction.py pode carregar esse antigo; convém treinar de novo e sobrescrever, ou guardar como outro nome.
# DistilBERT não usa token_type_ids, mas o teu código não lhes passa isso, por isso está tudo bem.

SEED = 42
MODEL_NAME = "bert-base-uncased"   # "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 2 # 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def move_batch_to_device(batch):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_loss_accuracy(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch)
            labels = batch["labels"]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def get_eval_outputs(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = outputs.logits.argmax(dim=1)

            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, cm


def main():
    df = get_datasets()

    df_train_full, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["Label"],
    )

    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=SEED,
        stratify=df_train_full["Label"],
    )

    y_train, label_map = encode_labels(df_train["Label"])
    y_val = np.array([label_map[label] for label in df_val["Label"]], dtype=np.int64)
    y_test = np.array([label_map[label] for label in df_test["Label"]], dtype=np.int64)
    n_classes = len(label_map)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = BertTextDataset(df_train["Text"].tolist(), y_train, tokenizer, MAX_LENGTH)
    val_dataset = BertTextDataset(df_val["Text"].tolist(), y_val, tokenizer, MAX_LENGTH)
    test_dataset = BertTextDataset(df_test["Text"].tolist(), y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
    ).to(device)

    classes = np.arange(n_classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=1,
        factor=0.5,
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    num_epochs_real = 0

    print(f"Training with model: {MODEL_NAME}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    print(f"Max tokens: {MAX_LENGTH} | Classes: {n_classes}")

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch)
            labels = batch["labels"]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate_loss_accuracy(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        num_epochs_real += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    epochs = range(1, num_epochs_real + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accuracies, marker="o", label="train acc")
    plt.plot(epochs, val_accuracies, marker="o", label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

    y_true, y_pred, cm = get_eval_outputs(model, test_loader)
    idx_to_label = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]

    print("Confusion matrix:\n", cm)
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=idx_to_label, digits=4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=idx_to_label)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.show()

    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    save_dir = module_path / "trained_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "bert_classifier.pth"

    print(f"Saving model to: {save_path}")
    torch.save(
        {
            "model_type": "bert",
            "model_name": MODEL_NAME,
            "model_state": model.state_dict(),
            "label_map": label_map,
            "max_length": MAX_LENGTH,
        },
        save_path,
    )
    print("Model saved.")
    print("Training complete.")


if __name__ == "__main__":
    main()
