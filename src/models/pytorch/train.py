import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich import box

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
        df = get_datasets(include_subm1=True)

        X_texts = df["Text"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        X_train, X_eval, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

        vector = build_vectorizer()
        vector.fit(X_train)
        X_train = vector.transform(X_train)
        X_eval = vector.transform(X_eval)

        train_dataset = TextDataset(X_train, y_train)
        eval_dataset = TextDataset(X_eval, y_eval)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        model = PyTorchModel.create(input_dim=X_train.shape[1], label_map=label_map, vector=vector)

        criterion = nn.CrossEntropyLoss()

        optimizer = Adam(model.model.parameters(), lr=learning_rate)
        writer = SummaryWriter()
        early_stop = EarlyStopping()
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        console = Console()
        progress = Progress(
            TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[epochs_total]}"),
            BarColumn(bar_width=30),
            TextColumn(
                "loss=[yellow]{task.fields[train_loss]:.4f}[/] "
                "val_loss=[red]{task.fields[val_loss]:.4f}[/] "
                "train=[green]{task.fields[train_acc]:.4f}[/] "
                "val=[magenta]{task.fields[val_acc]:.4f}[/]"
            ),
            TimeElapsedColumn(),
            console=console,
        )
        epoch_task = progress.add_task(
            "training",
            total=epochs,
            epoch=0,
            epochs_total=epochs,
            train_loss=0.0,
            val_loss=0.0,
            train_acc=0.0,
            val_acc=0.0,
        )

        with Live(progress, console=console, refresh_per_second=4):
            for epoch in range(epochs):
                model.model.train()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                for batch in train_loader:
                    X = batch["encoded"].to(torch_utils.device)
                    y = batch["labels"].to(torch_utils.device)

                    optimizer.zero_grad()
                    outputs = model.model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == y).sum().item()
                    total_samples += y.size(0)

                avg_train_loss = total_loss / len(train_loader)
                train_acc = total_correct / total_samples

                model.model.eval()
                eval_loss = 0.0
                eval_correct = 0
                eval_samples = 0
                all_preds = []

                with torch.no_grad():
                    for batch in eval_loader:
                        X = batch["encoded"].to(torch_utils.device)
                        y = batch["labels"].to(torch_utils.device)

                        outputs = model.model(X)
                        loss = criterion(outputs, y)
                        preds = outputs.argmax(dim=1)

                        eval_loss += loss.item()
                        eval_correct += (preds == y).sum().item()
                        eval_samples += y.size(0)
                        all_preds.extend(preds.cpu().tolist())

                avg_val_loss = eval_loss / len(eval_loader)
                val_acc = eval_correct / eval_samples

                inv_label_map = {v: k for k, v in label_map.items()}
                pred_labels = [inv_label_map[p] for p in all_preds]
                y_eval_labels = [inv_label_map[y] for y in y_eval]
                metrics = compute_metrics(pred_labels, y_eval_labels)

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)

                writer.add_scalar("Loss/train", avg_train_loss, epoch)
                writer.add_scalar("Loss/val", avg_val_loss, epoch)
                writer.add_scalar("Accuracy/train", train_acc, epoch)
                writer.add_scalar("Accuracy/test", val_acc, epoch)

                progress.update(
                    epoch_task,
                    advance=1,
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                )

                if early_stop.step(metrics["accuracy"]):
                    console.print("[bold yellow]Early stopping[/] triggered")
                    break

        torch.save({
            "model_type": type(model.model).__name__,
            "model_state": model.model.state_dict(),
            "label_map": label_map,
            "input_dim": X_train.shape[1],
            "vector": vector,
        }, main_folder / "pytorch-dnn.pt")

        final_preds = np.array(all_preds, dtype=np.int64)
        final_true = np.array(y_eval, dtype=np.int64)
        n_classes = len(unique_labels)
        final_per_class = np.zeros(n_classes, dtype=np.float32)
        for class_index in range(n_classes):
            class_mask = final_true == class_index
            if np.any(class_mask):
                final_per_class[class_index] = np.mean(final_preds[class_mask] == final_true[class_mask])

        results_table = Table(title="Final Results", box=box.ROUNDED)
        results_table.add_column("Class", style="cyan")
        results_table.add_column("Val Accuracy", justify="right")
        for label, acc in zip(unique_labels, final_per_class):
            results_table.add_row(label, f"{acc:.4f}")
        results_table.add_section()
        results_table.add_row("[bold]Overall[/]", f"[bold]{val_acc:.4f}[/]")
        console.print(results_table)

        return history

    @staticmethod
    def plot_history(history: dict):
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["train_loss"], label="train", color="#C44E52", linewidth=2)
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], label="train", color="#4C72B0", linewidth=2)
        ax2.plot(epochs, history["val_acc"], label="val", color="#55A868", linewidth=2, linestyle="--")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("PyTorch DNN — Training History", fontweight="bold")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pipeline():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        col1 = [
            ("", "", "#4C72B0"),
            ("Load Dataset", "get_datasets(include_subm1=True)", "#4C72B0"),
            ("Split Data", "train / validation", "#8C8C8C"),
            ("Build Vectorizer", "fit TF-IDF on train", "#DD8452"),
            ("Transform Text", "X_train and X_eval", "#DD8452"),
            ("Create Datasets", "TextDataset objects", "#8C8C8C"),
            ("Build DataLoaders", "mini-batches", "#8C8C8C"),
        ]

        col2 = [
            ("", "", "#55A868"),
            ("Build Model", "PyTorch DNN", "#55A868"),
            ("Forward Pass", "compute logits", "#55A868"),
            ("Loss", "CrossEntropyLoss", "#55A868"),
            ("Backward Pass", "loss.backward()", "#55A868"),
            ("Optimizer Step", "Adam update", "#55A868"),
            ("Validation", "loss + accuracy", "#55A868"),
            ("Early Stopping", "stop on no improvement", "#C44E52"),
            ("Save Model", "pytorch-dnn.pt", "#937860"),
        ]

        box_w, box_h, gap = 2.55, 0.68, 0.42
        col_sep = 0.6
        n_rows = max(len(col1), len(col2))
        total_h = n_rows * (box_h + gap) - gap

        fig, ax = plt.subplots(figsize=(8.0, total_h + 1.0))
        ax.axis("off")

        def draw_column(steps, x_offset, title):
            ax.text(
                x_offset + box_w / 2,
                total_h + box_h * 0.65,
                title,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

            for index, (name, detail, color) in enumerate(steps):
                y = total_h - index * (box_h + gap)
                if index > 0:
                    ax.annotate(
                        "",
                        xy=(x_offset + box_w / 2, y + box_h),
                        xytext=(x_offset + box_w / 2, y + box_h + gap),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
                    )

                rect = mpatches.FancyBboxPatch(
                    (x_offset, y),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.05",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.9,
                )
                ax.add_patch(rect)
                ax.text(
                    x_offset + box_w / 2,
                    y + box_h / 2 + 0.10,
                    name,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white",
                )
                ax.text(
                    x_offset + box_w / 2,
                    y + box_h / 2 - 0.15,
                    detail,
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="white",
                )

        draw_column(col1, 0.0, "① Data Preparation")
        draw_column(col2, box_w + col_sep, "② Training Loop")

        y_last_col1 = total_h - (len(col1) - 1) * (box_h + gap)
        y_first_col2 = total_h
        ax.annotate(
            "",
            xy=(box_w + col_sep, y_first_col2 + box_h / 2),
            xytext=(box_w, y_last_col1 + box_h / 2),
            arrowprops=dict(
                arrowstyle="->",
                color="#C44E52",
                lw=2.0,
                connectionstyle="arc3,rad=-0.3",
            ),
        )

        ax.set_xlim(-0.2, 2 * box_w + col_sep + 0.2)
        ax.set_ylim(-0.4, total_h + box_h + 0.9)
        ax.set_title("PyTorch DNN — Training Pipeline", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    history = PyTorchTrainer.train()
    PyTorchTrainer.plot_history(history)
