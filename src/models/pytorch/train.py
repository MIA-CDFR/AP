import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from prepare.dataset import get_datasets
from models.pytorch.model import PyTorchModel
from utils.dataset import build_vectorizer
from utils.pytorch import torch_utils
from utils.train import compute_metrics
from utils.model import main_folder


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class TextDataset(Dataset):
    def __init__(self, vector_texts, labels):
        if hasattr(vector_texts, "toarray"):
            vector_texts = vector_texts.toarray()

        self.encoded = vector_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = torch.tensor(self.encoded[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "encoded": encoded,
            "labels": label,
        }


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best = None
        self.counter = 0

    def step(self, metric):
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
            return False

        else:
            self.counter += 1
            return self.counter >= self.patience


class PyTorchTrainer:
    @staticmethod
    def train(epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
        df = get_datasets()

        X_texts = df["Text"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        X_train, X_eval, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

        vector = build_vectorizer()
        vector.fit(X_train)
        X_train = vector.transform(X_train)

        train_dataset = TextDataset(X_train, y_train)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = PyTorchModel.create(input_dim=X_train.shape[1], label_map=label_map, vector=vector)

        criterion = nn.CrossEntropyLoss()

        optimizer = Adam(model.model.parameters(), lr=learning_rate)
        writer = SummaryWriter()
        early_stop = EarlyStopping()

        for epoch in range(epochs):
            model.model.train()
            total_loss = 0.0

            for batch in train_loader:
                X = batch["encoded"].to(torch_utils.device)
                y = batch["labels"].to(torch_utils.device)

                optimizer.zero_grad()
                outputs = model.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
    
            avg_loss = total_loss / len(train_loader)

            preds = model.predict(X_eval)

            inv_label_map = {v: k for k, v in label_map.items()}
            y_eval_labels = [inv_label_map[y] for y in y_eval]

            metrics = compute_metrics(preds, y_eval_labels)

            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/test", metrics["accuracy"], epoch)

            print(f"Epoch {epoch+1} | loss {avg_loss:.4f} | acc {metrics['accuracy']:.4f}")

            if early_stop.step(metrics["accuracy"]):
                print("Early stopping triggered")
                break

        torch.save({
            "model_type": type(model.model).__name__,
            "model_state": model.model.state_dict(),
            "label_map": label_map,
            "input_dim": X_train.shape[1],
            "vector": vector,
        }, main_folder / "pytorch-dnn.pt")

if __name__ == "__main__":
    PyTorchTrainer.train()
