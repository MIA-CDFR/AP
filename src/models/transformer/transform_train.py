import random
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import torch

from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from prepare.dataset import get_datasets
from models.pytorch.prepare.feature import encode_labels, preprocess_text, preprocess_text_clean, build_vectorizer, build_handcrafted_matrix, standardize_train_test
from models.pytorch.prepare.dataset import TextDataset
from models.transformer.classifier.transformer import TransformerClassifier


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

from pathlib import Path
module_path = Path(__file__).resolve().parents[3]
BASELINE_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "subm1_labels_revealed.csv"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 40
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 1
USE_AMP = device.type == "cuda"

def normalize_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ").replace("\ufeff", " ")
    return " ".join(text.split()).strip()

def evaluate_loss_accuracy(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return avg_loss, acc, macro_f1, weighted_f1, y_true, y_pred

def get_confusion_matrix(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    return cm

def get_eval_outputs(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, cm

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

    #############################################################################################################################
    
    df = get_datasets()
    
    # Text normalization and deduplication for better generalization
    df = df.dropna(subset=["Text", "Label"]).copy()
    df["Text"] = df["Text"].astype(str).map(normalize_text)
    df = df[df["Text"].str.len() > 20]
    df = df.drop_duplicates(subset=["Text", "Label"]).reset_index(drop=True)
    print(f"After cleaning: {len(df)} samples")

    df_train, df_test, _, _ = train_test_split(
        df,
        df["Label"].to_numpy(),
        test_size=0.15,
        random_state=SEED,
        stratify=df["Label"].to_numpy()
    )

    # Keep raw texts for handcrafted features before cleaning
    raw_train = df_train["Text"].tolist()
    raw_test = df_test["Text"].tolist()

    # TF-IDF uses aggressive cleaning
    df_train["Text_tfidf"] = df_train["Text"].apply(preprocess_text)
    df_test["Text_tfidf"] = df_test["Text"].apply(preprocess_text)

    # Handcrafted features use light cleaning (preserves structure)
    clean_train = [preprocess_text_clean(t) for t in raw_train]
    clean_test = [preprocess_text_clean(t) for t in raw_test]

    # --- TF-IDF features ---
    vectorizer = build_vectorizer()
    X_tfidf_train = vectorizer.fit_transform(df_train["Text_tfidf"])
    X_tfidf_test = vectorizer.transform(df_test["Text_tfidf"])

    # --- Handcrafted features ---
    X_hand_train, feature_names = build_handcrafted_matrix(raw_train, clean_train)
    X_hand_test, _ = build_handcrafted_matrix(raw_test, clean_test)
    X_hand_train, X_hand_test, hand_mean, hand_std = standardize_train_test(X_hand_train, X_hand_test)

    #############################################################################################################################

    X_train = np.hstack([X_tfidf_train.toarray(), X_hand_train])
    X_test = np.hstack([X_tfidf_test.toarray(), X_hand_test])

    # --- NORMALIZAÇÃO ---
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # --- PSEUDO-SEQUÊNCIA ---
    seq_len = 32
    flat_input_dim = X_train.shape[1]

    pad_size = (seq_len - (flat_input_dim % seq_len)) % seq_len
    if pad_size > 0:
        X_train = np.hstack([X_train, np.zeros((X_train.shape[0], pad_size))])
        X_test = np.hstack([X_test, np.zeros((X_test.shape[0], pad_size))])

    new_input_dim = X_train.shape[1]
    embed_dim = new_input_dim // seq_len

    X_train = X_train.reshape(-1, seq_len, embed_dim)
    X_test = X_test.reshape(-1, seq_len, embed_dim)
    seq_len_model = X_train.shape[1]
    input_dim = X_train.shape[2]

    #############################################################################################################################
    
    print(f"TF-IDF features:      {X_tfidf_train.shape[1]}")
    print(f"Handcrafted features: {X_hand_train.shape[1]} {feature_names}")
    print(f"Sequence length: {X_train.shape[1]}, Embed dim: {X_train.shape[2]}")

    y_train, label_map = encode_labels(df_train["Label"])
    y_test = np.array([label_map[l] for l in df_test["Label"]])

    n_classes = len(label_map)

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_full_dataset = TextDataset(X_train, y_train)

    train_size = int(0.85 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        train_full_dataset, [train_size, val_size], generator=generator
    )
    
    # Balanced sampling for better class handling
    class_counts = np.bincount(y_train, minlength=n_classes)
    inv_freq = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = inv_freq[y_train]
    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    input_dim = X_train.shape[2]

    # model = DNNClassifier(input_dim, n_classes).to(device)
    # model = GRUClassifier(input_dim, n_classes).to(device)
    # model = LinearClassifier(input_dim, n_classes).to(device)
    # model = LogisticRegression(input_dim, n_classes).to(device)
    # model = LSTMClassifier(input_dim, n_classes).to(device)
    model = TransformerClassifier(input_dim, n_classes, seq_len=seq_len_model).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    
    # Grouped AdamW: different weight decay for different param groups
    no_decay = ["bias", "norm", "LayerNorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # Cosine scheduler with warmup for better convergence
    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(WARMUP_RATIO * num_training_steps)),
        num_training_steps=num_training_steps
    )

    #############################################################################################################################

    num_epochs_real = 0
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_state = None

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1_macros = []

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        optimizer.zero_grad(set_to_none=True)

        step = 0
        for xb, yb in train_loader:
            step += 1

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += (loss.item() * GRADIENT_ACCUMULATION_STEPS) * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += yb.size(0)

        # Flush remaining batch
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate_loss_accuracy(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_f1_macros.append(val_macro_f1)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_f1_macro={val_macro_f1:.4f} | val_f1_weighted={val_weighted_f1:.4f}"
        )

        num_epochs_real += 1
        
        # Checkpoint based on F1 macro (primary) then loss (secondary)
        is_best = (val_macro_f1 > best_val_f1 + 1e-4) or (
            abs(val_macro_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss
        )
        
        if is_best:
            best_val_f1 = val_macro_f1
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    epochs = range(1, num_epochs_real + 1)

    #############################################################################################################################

    # Gráficos
    epochs = range(1, num_epochs_real + 1)
    
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_accuracies, marker="o", label="train acc")
    plt.plot(epochs, val_accuracies, marker="o", label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transformer Training and Validation Accuracy")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.plot(epochs, val_f1_macros, marker="o", label="val f1 macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.title("Transformer Validation Macro-F1")
    plt.legend()
    plt.show()

    #############################################################################################################################

    cm = get_confusion_matrix(model, test_loader, device)
    print("Confusion matrix:\n", cm)

    y_true, y_pred, cm = get_eval_outputs(model, test_loader, device)

    idx_to_label = [k for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]
    
    test_macro_f1 = f1_score(y_true, y_pred, average="macro")
    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n=== Test metrics (never seen during training) ===")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=idx_to_label, digits=4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=idx_to_label)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.show()

    #############################################################################################################################

    # Gravar o modelo

    save_dir = module_path / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "model_type": "transformer",
        "model_state": model.state_dict(),
        "label_map": label_map,
        "vectorizer": vectorizer,
        "input_dim": input_dim,
        "seq_len": seq_len_model,
        "n_hand_features": X_hand_train.shape[1],
        "hand_feature_names": feature_names,
        "hand_mean": hand_mean,
        "hand_std": hand_std,
        "val_f1_macro": best_val_f1,
    }
    
    print(f"Saving full-precision model to: {save_dir / 'transformer.pth'}")
    torch.save(checkpoint_data, save_dir / "transformer.pth")
    print("  -> Full-precision checkpoint saved")
    
    save_fp16_checkpoint(
        model.state_dict(),
        {k: v for k, v in checkpoint_data.items() if k != "model_state"},
        save_dir / "transformer-fp16.pth"
    )

    print("Models saved.")

    #############################################################################################################################

    print("Training complete.")


if __name__ == "__main__":
    main()
