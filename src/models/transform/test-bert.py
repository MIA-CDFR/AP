import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import optim
from torch.utils.data import Dataset, DataLoader

from prepare.dataset import get_datasets


# =====================
# CONFIG
# =====================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
save_dir = module_path / "trained_models"
save_dir.mkdir(parents=True, exist_ok=True)


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def evaluate_bert(model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)

                total_loss += loss.item() * input_ids.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

        
def main():
    print("Start training")
    # =====================
    # DATA (ADAPT TO YOUR CSV)
    # =====================
    df = get_datasets()

    #Split
    # data = df.to_dict(orient="records")

    # bert_full_train_data = data[:2400]
    # bert_test_data = data[2400:3200]

    # bert_texts = [x["Text"] for x in bert_full_train_data]
    # bert_labels = [x["Label"] for x in bert_full_train_data]

    # bert_test_texts = [x["Text"] for x in bert_test_data]
    # bert_test_labels = [x["Label"] for x in bert_test_data]

    # split_idx = int(0.8 * len(bert_texts))

    # bert_train_texts = bert_texts[:split_idx]
    # bert_train_labels = bert_labels[:split_idx]

    # bert_val_texts = bert_texts[split_idx:]
    # bert_val_labels = bert_labels[split_idx:]

    # X_train = bert_train_texts
    # y_train = bert_train_labels

    # X_val = bert_val_texts
    # y_val = bert_val_labels

    # X_test = bert_test_texts
    # y_test = bert_test_labels

    data = df.to_dict(orient="records")

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=[x["Label"] for x in data]
    )

    train_texts = [x["Text"] for x in train_data]
    train_labels = [x["Label"] for x in train_data]

    test_texts = [x["Text"] for x in test_data]
    test_labels = [x["Label"] for x in test_data]

    # validação
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=42,
        stratify=train_labels
    )

    X_test = test_texts
    y_test = test_labels

    # must contain columns: Text, Label
    # texts = df["Text"].astype(str).tolist()
    # labels = df["Label"].tolist()

    # encode labels 
    labels = df["Label"].tolist()

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    #y = np.array([label_map[l] for l in labels])

    # Encode split labels to integer ids for torch tensors/loss.
    y_train_enc = np.array([label_map[l] for l in y_train], dtype=np.int64)
    y_val_enc = np.array([label_map[l] for l in y_val], dtype=np.int64)
    y_test_enc = np.array([label_map[l] for l in y_test], dtype=np.int64)

    # split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     texts, y, test_size=0.2, random_state=42, stratify=y
    # )

    n_classes = len(label_map)

    # =====================
    # TOKENIZER + MODEL
    # =====================
    #checkpoint = "distilbert/distilbert-base-uncased"
    checkpoint = "roberta-base"
    #roberta-base
    #checkpoint = "roberta-base"
    #checkpoint = "bert-base-uncased"
    #checkpoint = "bert-base-cased"
    #checkpoint = "microsoft/deberta-v3-base"#pip install transformers[sentencepiece]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=n_classes
    ).to(device)

    # #Tokenização
    # def bert_tokenize(texts, tokenizer, max_length=256):
    #     return tokenizer(
    #         texts,
    #         truncation=True,
    #         padding=True,
    #         max_length=max_length,
    #         return_tensors="pt",
    #     )

    # # complete
    # train_encodings = bert_tokenize(X_train, tokenizer)
    # val_encodings = bert_tokenize(X_val, tokenizer)
    # test_encodings = bert_tokenize(X_test, tokenizer)





    # =====================
    # DATASET
    # =====================
    # from torch.utils.data import TensorDataset

    # train_dataset = TensorDataset(
    #     train_encodings["input_ids"],
    #     train_encodings["attention_mask"],
    #     torch.tensor(y_train, dtype=torch.long),
    # )

    # val_dataset = TensorDataset(
    #     val_encodings["input_ids"],
    #     val_encodings["attention_mask"],
    #     torch.tensor(y_val, dtype=torch.long),
    # )

    # test_dataset = TensorDataset(
    #     test_encodings["input_ids"],
    #     test_encodings["attention_mask"],
    #     torch.tensor(y_test, dtype=torch.long),
    # )

    train_dataset = BERTDataset(X_train, y_train_enc, tokenizer, max_len=256)
    val_dataset = BERTDataset(X_val, y_val_enc, tokenizer, max_len=256)
    test_dataset = BERTDataset(X_test, y_test_enc, tokenizer, max_len=256)

    pin_memory = device.type == "cuda"
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=pin_memory)

    # =====================
    # LOSS + OPTIMIZER
    # =====================
    num_epochs = 3

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    from transformers import get_linear_schedule_with_warmup

    num_training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    # =====================
    # TRAIN
    # =====================
    

    bert_train_losses = []
    bert_val_losses = []
    bert_train_accuracies = []
    bert_val_accuracies = []

    best_val_loss = float("inf")
    patience = 2
    counter = 0

    from torch.amp import autocast, GradScaler

    scaler = GradScaler(device=device.type)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if True:
                with autocast(device_type=device.type):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                loss = criterion(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

            scheduler.step()

            # métricas
            running_loss += loss.item() * input_ids.size(0)

            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate_bert(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "checkpoint": checkpoint,
                "val_loss": val_loss,
                "epoch": epoch
            }, save_dir / "best_model.pth")

        else:
            counter += 1

        if counter >= patience:
            print("Early stopping!")
            break

        bert_train_losses.append(train_loss)
        bert_val_losses.append(val_loss)
        bert_train_accuracies.append(train_acc)
        bert_val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, bert_train_losses, marker="o", label="train loss")
    plt.plot(epochs, bert_val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DistilBERT Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, bert_train_accuracies, marker="o", label="train acc")
    plt.plot(epochs, bert_val_accuracies, marker="o", label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("DistilBERT Training and Validation Accuracy")
    plt.legend()
    plt.show()

    # =====================
    # EVALUATION
    # =====================
    checkpoint_data = torch.load(save_dir / "best_model.pth")
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("F1 macro:", f1_score(y_true, y_pred, average="macro"))
    print("F1 weighted:", f1_score(y_true, y_pred, average="weighted"))

    idx_to_label = {v: k for k, v in label_map.items()}

    y_true_labels = [idx_to_label[i] for i in y_true]
    y_pred_labels = [idx_to_label[i] for i in y_pred]

    #print(classification_report(y_true_labels, y_pred_labels))
    print("=== Numeric ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Labels ===")
    print(classification_report(y_true_labels, y_pred_labels))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    import seaborn as sns

    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


    # =====================
    # SAVE MODEL
    # =====================
    # module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    # save_dir = module_path / "trained_models"
    # save_dir.mkdir(parents=True, exist_ok=True)

    # val_loss, val_acc = evaluate_bert(model, val_loader, criterion)

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     counter = 0

    #     torch.save({
    #         "model_state": model.state_dict(),
    #         "label_map": label_map,
    #         "checkpoint": checkpoint,
    #         "val_loss": val_loss,
    #         "epoch": epoch
    #     }, save_dir / "bert_model.pth")

    # else:
    #     counter += 1

    print("Model saved!")

if __name__ == "__main__":
    main()
