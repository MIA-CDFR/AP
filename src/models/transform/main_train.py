import random
import numpy as np

import matplotlib.pyplot as plt
import torch

from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from prepare.dataset import get_datasets
from models.pytorch.prepare.feature import encode_labels, preprocess_text, preprocess_text_clean, build_vectorizer, build_handcrafted_matrix, standardize_train_test
from models.pytorch.prepare.dataset import TextDataset
from models.transform.transformers import TransformerClassifier


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate_loss_accuracy(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

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

def main():

    #############################################################################################################################
    
    df = get_datasets()

    df_train, df_test, _, _ = train_test_split(
        df,
        df["Label"].to_numpy(),
        test_size=0.2,
        random_state=42,
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_full_dataset = TextDataset(X_train, y_train)

    train_size = int(0.8 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    input_dim = X_train.shape[2]

    # model = DNNClassifier(input_dim, n_classes).to(device)
    # model = GRUClassifier(input_dim, n_classes).to(device)
    # model = LinearClassifier(input_dim, n_classes).to(device)
    # model = LogisticRegression(input_dim, n_classes).to(device)
    # model = LSTMClassifier(input_dim, n_classes).to(device)
    model = TransformerClassifier(input_dim, n_classes, seq_len=seq_len_model).to(device)

    classes = np.arange(n_classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    #############################################################################################################################

    num_epochs = 30
    num_epochs_real = 0
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # complete
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += yb.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate_loss_accuracy(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        num_epochs_real += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    epochs = range(1, num_epochs_real + 1)

    #############################################################################################################################

    # Gráficos
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_accuracies, marker="o", label="train acc")
    plt.plot(epochs, val_accuracies, marker="o", label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

    #############################################################################################################################

    cm = get_confusion_matrix(model, test_loader, device)
    print("Confusion matrix:\n", cm)

    y_true, y_pred, cm = get_eval_outputs(model, test_loader, device)

    idx_to_label = [k for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]

    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=idx_to_label, digits=4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=idx_to_label)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.show()

    #############################################################################################################################

    from pathlib import Path

    # Gravar o modelo

    module_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    save_dir = module_path / "trained_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to: {save_dir / 'transformer.pth'}")

    torch.save({
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
    }, save_dir  / "transformer.pth")

    print("Model saved.")

    #############################################################################################################################

    print("Training complete.")

if __name__ == "__main__":
    main()   
