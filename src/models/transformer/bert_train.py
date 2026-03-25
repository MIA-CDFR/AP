import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import math
from collections import Counter

from contextlib import nullcontext
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from torch import optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from prepare.dataset import get_datasets


# =====================
# CONFIG
# =====================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

module_path = Path(__file__).resolve().parents[3]
save_dir = module_path / "models"
save_dir.mkdir(parents=True, exist_ok=True)

BASELINE_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "subm1_labels_revealed.csv"

MAX_LEN = 256
BATCH_SIZE = 12
NUM_EPOCHS = 10
PATIENCE = 4
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 2
USE_AMP = device.type == "cuda"
MAX_CHAR_LEN = 6000


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

def normalize_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ").replace("\ufeff", " ")
    return " ".join(text.split()).strip()


def print_data_profile(df: pd.DataFrame, tokenizer=None, max_len: int = 256, title: str = "Profile"):
    print(f"\n=== {title} ===")
    print(f"Samples: {len(df)}")
    print("Label counts:")
    print(df["Label"].value_counts().to_string())

    char_len = df["Text"].astype(str).str.len()
    print(
        "Char length stats: "
        f"median={int(char_len.median())}, "
        f"p90={int(char_len.quantile(0.90))}, "
        f"p95={int(char_len.quantile(0.95))}, "
        f"p99={int(char_len.quantile(0.99))}, "
        f"max={int(char_len.max())}"
    )

    if tokenizer is not None:
        per_class_total = Counter()
        per_class_trunc = Counter()

        for text, label in zip(df["Text"].tolist(), df["Label"].tolist()):
            token_ids = tokenizer.encode(str(text), add_special_tokens=True, truncation=False)
            per_class_total[label] += 1
            if len(token_ids) > max_len:
                per_class_trunc[label] += 1

        print(f"Token truncation rate by class (max_len={max_len}):")
        for label, total in sorted(per_class_total.items(), key=lambda x: x[0]):
            rate = per_class_trunc[label] / max(1, total)
            print(f"  {label}: {rate:.2%} ({per_class_trunc[label]}/{total})")


def evaluate_bert(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

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

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return total_loss / total, correct / total, macro_f1, weighted_f1, y_true, y_pred


def evaluate_on_baseline_csv(model, tokenizer, label_map):
    if not BASELINE_CSV_PATH.exists():
        print(f"Baseline CSV not found at: {BASELINE_CSV_PATH}")
        return

    baseline_df = pd.read_csv(BASELINE_CSV_PATH, sep=";")
    baseline_df.columns = [c.strip() for c in baseline_df.columns]
    baseline_df = baseline_df.dropna(subset=["Text", "Label"]).copy()
    baseline_df["Text"] = baseline_df["Text"].astype(str).map(normalize_text)
    baseline_df = baseline_df[baseline_df["Text"].str.len() > 20]

    known_mask = baseline_df["Label"].isin(set(label_map.keys()))
    baseline_df = baseline_df[known_mask]

    if baseline_df.empty:
        print("Baseline CSV has no labels compatible with current label_map.")
        return

    baseline_texts = baseline_df["Text"].tolist()
    baseline_labels = baseline_df["Label"].map(label_map).astype(int).tolist()

    baseline_dataset = BERTDataset(baseline_texts, baseline_labels, tokenizer, max_len=MAX_LEN)
    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    criterion = nn.CrossEntropyLoss()
    baseline_loss, baseline_acc, baseline_macro_f1, baseline_weighted_f1, _, _ = evaluate_bert(
        model, baseline_loader, criterion
    )

    print("\n=== Baseline-only evaluation (subm1_labels_revealed.csv) ===")
    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"Baseline acc: {baseline_acc:.4f}")
    print(f"Baseline F1 macro: {baseline_macro_f1:.4f}")
    print(f"Baseline F1 weighted: {baseline_weighted_f1:.4f}")


def save_fp16_checkpoint(model_state, metadata, save_path):
    """
    Save model checkpoint in FP16 format to reduce file size (~50% smaller).
    Non-floating-point tensors are preserved as-is.
    """
    fp16_state = {
        k: (v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu())
        for k, v in model_state.items()
    }
    torch.save(
        {**metadata, "model_state": fp16_state},
        save_path
    )
    original_size = sum(v.numel() * v.element_size() for v in model_state.values()) / (1024**2)
    compressed_size = sum(v.numel() * v.element_size() for v in fp16_state.values()) / (1024**2)
    print(f"  -> FP16 checkpoint: {compressed_size:.1f}MB (orig: {original_size:.1f}MB, ~{(1-compressed_size/original_size)*100:.0f}% savings)")

        
def main():
    print("Start training")
    df = get_datasets()
    df = df.dropna(subset=["Text", "Label"]).copy()
    df["Text"] = df["Text"].astype(str).map(normalize_text)
    df = df[df["Text"].str.len() > 20]
    df = df[df["Text"].str.len() <= MAX_CHAR_LEN]
    df = df.drop_duplicates(subset=["Text", "Label"]).reset_index(drop=True)

    data = df.to_dict(orient="records")

    train_data, test_data = train_test_split(
        data,
        test_size=0.15,
        random_state=SEED,
        stratify=[x["Label"] for x in data]
    )

    train_texts = [x["Text"] for x in train_data]
    train_labels = [x["Label"] for x in train_data]

    test_texts = [x["Text"] for x in test_data]
    test_labels = [x["Label"] for x in test_data]

    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        train_labels,
        test_size=0.15,
        random_state=SEED,
        stratify=train_labels
    )

    X_test = test_texts
    y_test = test_labels

    labels = df["Label"].tolist()

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}

    y_train_enc = np.array([label_map[l] for l in y_train], dtype=np.int64)
    y_val_enc = np.array([label_map[l] for l in y_val], dtype=np.int64)
    y_test_enc = np.array([label_map[l] for l in y_test], dtype=np.int64)

    n_classes = len(label_map)
    print(f"Samples: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print("Train class distribution:", {k: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})

    checkpoint = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print_data_profile(df, tokenizer=tokenizer, max_len=MAX_LEN, title="Training corpus profile")

    model_config = AutoConfig.from_pretrained(
        checkpoint,
        num_labels=n_classes,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.2,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        config=model_config
    ).to(device)

    train_dataset = BERTDataset(X_train, y_train_enc, tokenizer, max_len=MAX_LEN)
    val_dataset = BERTDataset(X_val, y_val_enc, tokenizer, max_len=MAX_LEN)
    test_dataset = BERTDataset(X_test, y_test_enc, tokenizer, max_len=MAX_LEN)

    class_counts = np.bincount(y_train_enc, minlength=n_classes)
    inv_freq = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = inv_freq[y_train_enc]
    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    pin_memory = device.type == "cuda"
    num_workers = 0 if device.type == "mps" else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_weights = 1.0 / np.clip(class_counts, 1, None)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    from transformers import get_cosine_schedule_with_warmup

    num_training_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(WARMUP_RATIO * num_training_steps)),
        num_training_steps=num_training_steps
    )

    bert_train_losses = []
    bert_val_losses = []
    bert_train_accuracies = []
    bert_val_accuracies = []
    bert_val_f1_macro = []

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    counter = 0

    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if USE_AMP
                else nullcontext()
            )

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += (loss.item() * GRADIENT_ACCUMULATION_STEPS) * input_ids.size(0)

            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate_bert(model, val_loader, criterion)

        is_best = (val_macro_f1 > best_val_f1 + 1e-4) or (
            abs(val_macro_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss
        )

        if is_best:
            best_val_f1 = val_macro_f1
            best_val_loss = val_loss
            counter = 0

            checkpoint_data = {
                "model_state": model.state_dict(),
                "label_map": label_map,
                "model_name": checkpoint,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "epoch": epoch
            }
            
            torch.save(checkpoint_data, save_dir / "transformer-bert.pth")
            print(f"  -> Full-precision checkpoint saved")
            
            save_fp16_checkpoint(
                checkpoint_data["model_state"],
                {k: v for k, v in checkpoint_data.items() if k != "model_state"},
                save_dir / "transformer-bert-fp16.pth"
            )

        else:
            counter += 1

        bert_train_losses.append(train_loss)
        bert_val_losses.append(val_loss)
        bert_train_accuracies.append(train_acc)
        bert_val_accuracies.append(val_acc)
        bert_val_f1_macro.append(val_macro_f1)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_f1_macro={val_macro_f1:.4f} | val_f1_weighted={val_weighted_f1:.4f}"
        )

        if counter >= PATIENCE:
            print("Early stopping!")
            break

    epochs = range(1, len(bert_train_losses) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, bert_train_losses, marker="o", label="train loss")
    plt.plot(epochs, bert_val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RoBERTa Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, bert_train_accuracies, marker="o", label="train acc")
    plt.plot(epochs, bert_val_accuracies, marker="o", label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("RoBERTa Training and Validation Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, bert_val_f1_macro, marker="o", label="val f1 macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.title("RoBERTa Validation Macro-F1")
    plt.legend()
    plt.show()

    # =====================
    # EVALUATION
    # =====================
    checkpoint_data = torch.load(save_dir / "transformer-bert.pth", map_location=device)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()

    test_loss, test_acc, test_macro_f1, test_weighted_f1, y_true, y_pred = evaluate_bert(
        model, test_loader, nn.CrossEntropyLoss()
    )

    print("\n=== Test metrics (never seen during training) ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 macro: {test_macro_f1:.4f}")
    print(f"F1 weighted: {test_weighted_f1:.4f}")

    idx_to_label = {v: k for k, v in label_map.items()}

    y_true_labels = [idx_to_label[i] for i in y_true]
    y_pred_labels = [idx_to_label[i] for i in y_pred]

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

    evaluate_on_baseline_csv(model, tokenizer, label_map)

    print("Model saved!")

if __name__ == "__main__":
    main()
