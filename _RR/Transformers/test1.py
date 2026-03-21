#https://medium.com/@piyushkashyap045/guide-to-tokenization-and-padding-with-bert-transforming-text-into-machine-readable-data-5a24bf59d36b
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

# ============================================================
# 1. PREPROCESSAMENTO
# ============================================================

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# 2. DATASET
# ============================================================

def load_data():
    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", name="in_domain")

    df_train = dataset["train"].to_pandas().sample(30000, random_state=42)
    df_test = dataset["test"].to_pandas().sample(8000, random_state=42)

    df_train.rename(columns={"content": "Text", "model": "Label"}, inplace=True)
    df_test.rename(columns={"content": "Text", "model": "Label"}, inplace=True)

    return df_train, df_test

# ============================================================
# 3. VECTORIZE (TF-IDF)
# ============================================================

def build_vectorizer():
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words="english"
    )

# ============================================================
# 4. DATASET CLASS
# ============================================================

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze()
        y = self.y[idx]
        return x, y

# ============================================================
# 5. MODELOS
# ============================================================

# ---------------- TRANSFORMER (from scratch - based on classes) ----------------

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, d_model=128, num_heads=4):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        x = x.mean(dim=1)
        return self.classifier(x)


# ---------------- BERT (pre-trained transformer) ----------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTWrapper(nn.Module):
    def __init__(self, n_classes, model_name="bert-base-uncased"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits


# ============================================================
# 6. TREINO (standard models)
# ============================================================
# ============================================================

class DNN(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class GRUModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        _, h = self.gru(x)
        return self.fc(h[-1])


class LSTMModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ============================================================
# 6. TREINO
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# ============================================================
# 7. MAIN
# ============================================================

def main():

    df_train, df_test = load_data()

    df_train["Text"] = df_train["Text"].apply(preprocess_text)
    df_test["Text"] = df_test["Text"].apply(preprocess_text)

    vectorizer = build_vectorizer()

    X_train = vectorizer.fit_transform(df_train["Text"])
    X_test = vectorizer.transform(df_test["Text"])

    labels = sorted(df_train["Label"].unique())
    label_map = {l:i for i,l in enumerate(labels)}

    y_train = np.array([label_map[l] for l in df_train["Label"]])
    y_test = np.array([label_map[l] for l in df_test["Label"]])

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    n_classes = len(label_map)

    # Escolher modelo aqui
    model = DNN(input_dim, n_classes).to(device)
    # model = GRUModel(input_dim, n_classes).to(device)
    # model = LSTMModel(input_dim, n_classes).to(device)
    # model = TransformerClassifier(input_dim, n_classes).to(device)  # NOVO

    # NOTA: BERT precisa pipeline diferente (tokenizer + input_ids)
    model = DNN(input_dim, n_classes).to(device)
    # model = GRUModel(input_dim, n_classes).to(device)
    # model = LSTMModel(input_dim, n_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | Acc {acc:.4f}")

    print("\nFinal accuracy:", acc)


# ============================================================
# 8. BERT PIPELINE (AVANÇADO - RECOMENDADO)
# ============================================================

from transformers import BertTokenizer

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_bert(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_bert(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def run_bert(df_train, df_test, label_map):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    y_train = np.array([label_map[l] for l in df_train["Label"]])
    y_test = np.array([label_map[l] for l in df_test["Label"]])

    train_dataset = BERTDataset(df_train["Text"].tolist(), y_train, tokenizer)
    test_dataset = BERTDataset(df_test["Text"].tolist(), y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BERTWrapper(n_classes=len(label_map), model_name="bert-base-uncased").to(device)

    # Podes testar outros modelos facilmente:
    # model = BERTWrapper(n_classes=len(label_map), model_name="distilbert-base-uncased").to(device)
    # model = BERTWrapper(n_classes=len(label_map), model_name="roberta-base").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print("===== BERT TRAINING =====")

    for epoch in range(3):
        loss = train_bert(model, train_loader, optimizer, device)
        acc = evaluate_bert(model, test_loader, device)

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | Acc {acc:.4f}")

    print("Final BERT accuracy:", acc)


# ============================================================
# MAIN (UPDATED)
# ============================================================

if __name__ == "__main__":

    df_train, df_test = load_data()

    df_train["Text"] = df_train["Text"].apply(preprocess_text)
    df_test["Text"] = df_test["Text"].apply(preprocess_text)

    labels = sorted(df_train["Label"].unique())
    label_map = {l:i for i,l in enumerate(labels)}

    # RUN CLASSICAL MODELS
    main()

    # RUN BERT (TOP MODEL)
    run_bert(df_train, df_test, label_map)

