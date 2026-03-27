import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from prepare.dataset import get_datasets
from models.transformer.model import TransformModel
from utils.pytorch import torch_utils
from utils.train import compute_metrics_logits
from utils.model import main_folder


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

seq_len: int = 512
model_name: str = "distilbert-base-uncased"


class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, seq_len):
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": label,
        }


class EpochMetricsCallback(TrainerCallback):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _compute_metrics(self, model, dataloader):
        model.eval()
        model_device = next(model.parameters()).device
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(model_device)
                attention_mask = batch["attention_mask"].to(model_device)
                labels = batch["labels"].to(model_device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else self.criterion(logits, labels)

                preds = logits.argmax(dim=1)
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / max(1, len(dataloader))
        acc = total_correct / max(1, total_samples)
        return avg_loss, acc

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        train_dataloader = kwargs.get("train_dataloader")
        eval_dataloader = kwargs.get("eval_dataloader")

        if model is None or train_dataloader is None or eval_dataloader is None:
            return

        train_loss, train_acc = self._compute_metrics(model, train_dataloader)
        val_loss, val_acc = self._compute_metrics(model, eval_dataloader)

        self.history["epoch"].append(float(state.epoch) if state.epoch is not None else len(self.history["epoch"]) + 1)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)


class TransformerTrainer:
    d_model: int = 512
    num_heads: int = 16
    num_layers: int = 6
    dropout: float = 0.2
    @staticmethod
    def train(epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-4, weight_decay: float = 0.01):
        df = get_datasets(include_subm1=True)

        X_texts = df["Text"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        X_train, X_eval, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = TextClsDataset(X_train, y_train, tokenizer, seq_len)
        eval_dataset = TextClsDataset(X_eval, y_eval, tokenizer, seq_len)

        model = TransformModel.create(
            label_map=label_map,
            tokenizer_name=model_name,
            vocab_size=tokenizer.vocab_size,
            pad_idx=tokenizer.pad_token_id or 0,
            seq_len=seq_len,
            d_model=TransformerTrainer.d_model,
            num_heads=TransformerTrainer.num_heads,
            num_layers=TransformerTrainer.num_layers,
            dropout=TransformerTrainer.dropout,
        )

        training_args = TrainingArguments(
            output_dir=main_folder / "results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=20,
            dataloader_num_workers=4 if torch_utils.device.type == "cpu" else 0,
            dataloader_pin_memory=torch_utils.device.type == "cuda",
            report_to="none",
            fp16=False,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            gradient_accumulation_steps=1,
        )

        metrics_callback = EpochMetricsCallback()

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_logits,
            callbacks=[metrics_callback],
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

        torch.save(
            {
                "model_state": model.model.state_dict(),
                "tokenizer_name": model_name,
                "pad_token_id": tokenizer.pad_token_id,
                "seq_len": seq_len,
                "label_map": label_map,
                "vocab_size": tokenizer.vocab_size,
                "d_model": TransformerTrainer.d_model,
                "num_heads": TransformerTrainer.num_heads,
                "num_layers": TransformerTrainer.num_layers,
                "dropout": TransformerTrainer.dropout,
            },
            main_folder / "transformer.pt",
        )
        print("Model saved to", main_folder / "transformer.pt")

        return metrics_callback.history

    @staticmethod
    def plot_history(history: dict):
        import matplotlib.pyplot as plt

        epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["train_loss"], label="train", color="#C44E52", linewidth=2)
        ax1.plot(epochs, history["val_loss"], label="test/val", color="#4C72B0", linewidth=2, linestyle="--")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], label="train", color="#55A868", linewidth=2)
        ax2.plot(epochs, history["val_acc"], label="test/val", color="#8172B2", linewidth=2, linestyle="--")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Transformer - Training History", fontweight="bold")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_architecture(seq_length: int = 512):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        layers = [
            ("Token IDs", f"[B, T]   T <= {seq_length}", "#4C72B0"),
            ("Embedding", f"token + positional  d_model={TransformerTrainer.d_model}", "#4C72B0"),
            (
                "Transformer Encoder",
                f"{TransformerTrainer.num_layers} blocks, {TransformerTrainer.num_heads} heads",
                "#55A868",
            ),
            ("Mean Pooling", "mask-aware pooling over tokens", "#8C8C8C"),
            ("Dropout", f"p={TransformerTrainer.dropout}", "#DD8452"),
            ("Linear Classifier", "logits -> n_classes", "#C44E52"),
        ]

        box_w, box_h, gap = 2.6, 0.72, 0.35
        total_h = len(layers) * (box_h + gap) - gap

        fig, ax = plt.subplots(figsize=(6.2, total_h + 0.9))
        ax.axis("off")

        for idx, (name, detail, color) in enumerate(layers):
            y = total_h - idx * (box_h + gap)
            if idx > 0:
                ax.annotate(
                    "",
                    xy=(box_w / 2, y + box_h),
                    xytext=(box_w / 2, y + box_h + gap),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6),
                )

            rect = mpatches.FancyBboxPatch(
                (0, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="white",
                linewidth=1.6,
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(
                box_w / 2,
                y + box_h / 2 + 0.10,
                name,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
            ax.text(
                box_w / 2,
                y + box_h / 2 - 0.14,
                detail,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
            )

        ax.set_xlim(-0.2, box_w + 0.2)
        ax.set_ylim(-0.5, total_h + box_h + 0.9)
        ax.set_title("Transformer Classifier - Architecture", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_paper_style_architecture(seq_length: int = 512):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(7.8, 9.2))
        ax.axis("off")

        def box(x, y, w, h, text, color, fontsize=10, lw=1.8, rounded=True):
            patch = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04" if rounded else "square,pad=0.02",
                facecolor=color,
                edgecolor="black",
                linewidth=lw,
            )
            ax.add_patch(patch)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
            return patch

        def arrow(x1, y1, x2, y2, lw=1.8):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw),
            )

        def encoder_block(x, y, w, h):
            outer = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.05",
                facecolor="#F7F7F7",
                edgecolor="black",
                linewidth=2.0,
            )
            ax.add_patch(outer)

            mha_y = y + 0.23 * h
            add1_y = y + 0.46 * h
            ffn_y = y + 0.63 * h
            add2_y = y + 0.84 * h

            box(x + 0.12 * w, mha_y, 0.76 * w, 0.16 * h, "Multi-Head\nAttention", "#FDE0B6", fontsize=9)
            box(x + 0.18 * w, add1_y, 0.64 * w, 0.09 * h, "Add & Norm", "#F7F1B5", fontsize=9)
            box(x + 0.12 * w, ffn_y, 0.76 * w, 0.16 * h, "Feed\nForward", "#B9DDF1", fontsize=9)
            box(x + 0.18 * w, add2_y, 0.64 * w, 0.09 * h, "Add & Norm", "#F7F1B5", fontsize=9)

            center_x = x + w / 2
            arrow(center_x, y + 0.02 * h, center_x, mha_y)
            arrow(center_x, mha_y + 0.16 * h, center_x, add1_y)
            arrow(center_x, add1_y + 0.09 * h, center_x, ffn_y)
            arrow(center_x, ffn_y + 0.16 * h, center_x, add2_y)
            arrow(center_x, add2_y + 0.09 * h, center_x, y + h + 0.02 * h)

            ax.annotate(
                "",
                xy=(x + 0.18 * w, add1_y + 0.045 * h),
                xytext=(x + 0.05 * w, y + 0.04 * h),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black", connectionstyle="arc3,rad=-0.6"),
            )
            ax.annotate(
                "",
                xy=(x + 0.18 * w, add2_y + 0.045 * h),
                xytext=(x + 0.05 * w, add1_y + 0.09 * h),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black", connectionstyle="arc3,rad=-0.6"),
            )

            ax.text(x + w + 0.18, y + h / 2, f"N× = {TransformerTrainer.num_layers}", fontsize=12, va="center")

        x_mid = 3.9
        block_w = 2.4
        block_h = 4.3
        base_y = 1.9

        box(x_mid - 0.7, 0.25, 1.4, 0.55, "Input\nEmbedding", "#F8D7DA", fontsize=10)
        ax.text(x_mid - 1.75, 1.03, "Positional\nEncoding", fontsize=11, ha="center")
        ax.text(x_mid, -0.3, "Input Tokens", fontsize=11, ha="center")
        arrow(x_mid, -0.09, x_mid, 0.25)

        add_circle = mpatches.Circle((x_mid, 1.3), 0.12, facecolor="white", edgecolor="black", linewidth=1.6)
        ax.add_patch(add_circle)
        ax.text(x_mid, 1.3, "+", ha="center", va="center", fontsize=14)
        pos_circle = mpatches.Circle((x_mid - 0.7, 1.3), 0.19, facecolor="white", edgecolor="black", linewidth=1.6)
        ax.add_patch(pos_circle)
        ax.add_patch(mpatches.Arc((x_mid - 0.76, 1.3), 0.22, 0.22, theta1=20, theta2=340, linewidth=1.3))
        ax.add_patch(mpatches.Arc((x_mid - 0.64, 1.3), 0.22, 0.22, theta1=200, theta2=160, linewidth=1.3))
        arrow(x_mid, 0.8, x_mid, 1.2)
        arrow(x_mid - 0.51, 1.3, x_mid - 0.10, 1.3)
        arrow(x_mid, 1.4, x_mid, base_y)

        encoder_block(x_mid - block_w / 2, base_y, block_w, block_h)

        box(x_mid - 0.8, base_y + block_h + 0.35, 1.6, 0.55, "Mean Pooling", "#D9D9D9", fontsize=10)
        box(x_mid - 0.8, base_y + block_h + 1.15, 1.6, 0.55, f"Dropout\np={TransformerTrainer.dropout}", "#FDE0B6", fontsize=10)
        box(x_mid - 0.8, base_y + block_h + 1.95, 1.6, 0.55, "Linear", "#D9E1F2", fontsize=11)
        box(x_mid - 0.8, base_y + block_h + 2.75, 1.6, 0.55, "Softmax", "#D9EAD3", fontsize=11)

        arrow(x_mid, base_y + block_h, x_mid, base_y + block_h + 0.35)
        arrow(x_mid, base_y + block_h + 0.90, x_mid, base_y + block_h + 1.15)
        arrow(x_mid, base_y + block_h + 1.70, x_mid, base_y + block_h + 1.95)
        arrow(x_mid, base_y + block_h + 2.50, x_mid, base_y + block_h + 2.75)
        arrow(x_mid, base_y + block_h + 3.30, x_mid, base_y + block_h + 3.70)

        ax.text(x_mid + 0.02, base_y + block_h + 4.0, "Class Probabilities", fontsize=12, ha="center")
        ax.text(
            x_mid,
            11,
            (
                "Encoder-only Transformer classifier\n"
                f"d_model={TransformerTrainer.d_model}, heads={TransformerTrainer.num_heads}, seq_len<={seq_length}"
            ),
            fontsize=12,
            ha="center",
            fontweight="bold",
        )

        ax.set_xlim(1.0, 6.8)
        ax.set_ylim(0, 10)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pipeline():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        col1 = [
            ("", "", "#4C72B0"),
            ("Load Dataset", "get_datasets()", "#4C72B0"),
            ("Split Data", "train / validation", "#8C8C8C"),
            ("Tokenizer", "AutoTokenizer.from_pretrained", "#4C72B0"),
            ("Build Datasets", "TextClsDataset(train, eval)", "#8C8C8C"),
            ("Create Model", "TransformModel.create(...)", "#55A868"),
        ]

        col2 = [
            ("", "", "#55A868"),
            ("Trainer Setup", "TrainingArguments + Trainer", "#55A868"),
            ("Train", "trainer.train()", "#55A868"),
            ("Evaluate", "trainer.evaluate()", "#55A868"),
            ("Select Best", "f1_macro + load_best_model", "#C44E52"),
            ("Save Checkpoint", "transformer.pt", "#937860"),
        ]

        box_w, box_h, gap = 2.7, 0.72, 0.42
        col_sep = 0.7
        n_rows = max(len(col1), len(col2))
        total_h = n_rows * (box_h + gap) - gap

        fig, ax = plt.subplots(figsize=(8.6, total_h + 1.0))
        ax.axis("off")

        def draw_column(steps, x_offset, title):
            ax.text(
                x_offset + box_w / 2,
                total_h + box_h * 0.70,
                title,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

            for idx, (name, detail, color) in enumerate(steps):
                y = total_h - idx * (box_h + gap)
                if idx > 0:
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
                    linewidth=1.6,
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
                    y + box_h / 2 - 0.14,
                    detail,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white",
                )

        draw_column(col1, 0.0, "1) Data and Model Setup")
        draw_column(col2, box_w + col_sep, "2) Optimization Loop")

        y_last_col1 = total_h - (len(col1) - 1) * (box_h + gap)
        y_first_col2 = total_h
        ax.annotate(
            "",
            xy=(box_w + col_sep, y_first_col2 + box_h / 2),
            xytext=(box_w, y_last_col1 + box_h / 2),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2.0, connectionstyle="arc3,rad=-0.3"),
        )

        ax.set_xlim(-0.2, 2 * box_w + col_sep + 0.2)
        ax.set_ylim(-0.5, total_h + box_h + 1.0)
        ax.set_title("Transformer - Training Pipeline", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    history = TransformerTrainer.train()
    TransformerTrainer.plot_history(history)
