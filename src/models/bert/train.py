import torch

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

from prepare.dataset import get_datasets
from models.bert.model import BertModel
from utils.pytorch import torch_utils
from utils.train import compute_metrics_logits
from utils.model import main_folder


MODEL_NAME = "roberta-base"

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BertTrainer:

    @staticmethod
    def plot_paper_transformer_architecture():
        """Draw a paper-style encoder-only Transformer architecture used by BERT/RoBERTa."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(7.0, 9.2))
        ax.axis("off")

        def draw_box(x, y, w, h, text, color, fs=10):
            rect = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="black",
                linewidth=1.8,
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
            return rect

        def draw_arrow(x1, y1, x2, y2, lw=1.8, rad=0.0):
            cs = "arc3,rad=0" if rad == 0 else f"arc3,rad={rad}"
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=lw, connectionstyle=cs),
            )

        def draw_positional_symbol(cx, cy, r=0.19):
            circle = mpatches.Circle((cx, cy), r, facecolor="white", edgecolor="black", linewidth=1.6)
            ax.add_patch(circle)
            ax.add_patch(mpatches.Arc((cx - 0.05, cy), 0.24, 0.24, theta1=20, theta2=340, linewidth=1.2))
            ax.add_patch(mpatches.Arc((cx + 0.05, cy), 0.24, 0.24, theta1=200, theta2=160, linewidth=1.2))

        def draw_plus(cx, cy, r=0.12):
            plus = mpatches.Circle((cx, cy), r, facecolor="white", edgecolor="black", linewidth=1.6)
            ax.add_patch(plus)
            ax.text(cx, cy, "+", ha="center", va="center", fontsize=14)

        def draw_encoder_block(x, y, w, h):
            outer = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.06",
                facecolor="#F7F7F7",
                edgecolor="black",
                linewidth=2.0,
            )
            ax.add_patch(outer)

            y_mha = y + 0.23 * h
            y_add1 = y + 0.46 * h
            y_ffn = y + 0.63 * h
            y_add2 = y + 0.84 * h

            draw_box(x + 0.12 * w, y_mha, 0.76 * w, 0.16 * h, "Multi-Head\nAttention", "#FDE0B6", 9)
            draw_box(x + 0.18 * w, y_add1, 0.64 * w, 0.09 * h, "Add & Norm", "#F7F1B5", 9)
            draw_box(x + 0.12 * w, y_ffn, 0.76 * w, 0.16 * h, "Feed\nForward", "#B9DDF1", 9)
            draw_box(x + 0.18 * w, y_add2, 0.64 * w, 0.09 * h, "Add & Norm", "#F7F1B5", 9)

            cx = x + w / 2
            draw_arrow(cx, y + 0.02 * h, cx, y_mha)
            draw_arrow(cx, y_mha + 0.16 * h, cx, y_add1)
            draw_arrow(cx, y_add1 + 0.09 * h, cx, y_ffn)
            draw_arrow(cx, y_ffn + 0.16 * h, cx, y_add2)
            draw_arrow(cx, y_add2 + 0.09 * h, cx, y + h - 0.02 * h)

            # residual paths
            draw_arrow(x + 0.05 * w, y + 0.05 * h, x + 0.18 * w, y_add1 + 0.045 * h, rad=-0.6)
            draw_arrow(x + 0.05 * w, y_add1 + 0.09 * h, x + 0.18 * w, y_add2 + 0.045 * h, rad=-0.6)

        # geometry (encoder-only)
        x_mid = 3.6
        base_y = 2.0
        block_w, block_h = 2.0, 3.9

        # input embedding + positional encoding
        draw_box(x_mid - 0.8, 0.6, 1.6, 0.6, "Input\nEmbedding", "#F8D7DA", 11)
        draw_plus(x_mid, 1.5)
        draw_positional_symbol(x_mid - 0.55, 1.5)

        ax.text(x_mid - 1.25, 1.5, "Positional\nEncoding", fontsize=12, ha="right", va="center")
        ax.text(x_mid, 0.0, "Inputs", fontsize=14, ha="center")

        draw_arrow(x_mid, 0.25, x_mid, 0.6)
        draw_arrow(x_mid, 1.2, x_mid, 1.33)
        draw_arrow(x_mid - 0.35, 1.5, x_mid - 0.02, 1.5)
        draw_arrow(x_mid, 1.62, x_mid, base_y)

        # encoder stack
        draw_encoder_block(x_mid - block_w / 2, base_y, block_w, block_h)
        ax.text(x_mid + block_w / 2 + 0.22, base_y + block_h / 2, "N×", fontsize=14, va="center")

        # classification head used by BERT/RoBERTa
        draw_box(x_mid - 0.95, base_y + block_h + 0.35, 1.9, 0.58, "[CLS] / Pooling", "#D9D9D9", 10)
        draw_box(x_mid - 0.95, base_y + block_h + 1.15, 1.9, 0.58, "Dropout + Linear", "#D9E1F2", 10)
        draw_box(x_mid - 0.95, base_y + block_h + 1.95, 1.9, 0.58, "Softmax", "#D9EAD3", 11)

        draw_arrow(x_mid, base_y + block_h, x_mid, base_y + block_h + 0.35)
        draw_arrow(x_mid, base_y + block_h + 0.93, x_mid, base_y + block_h + 1.15)
        draw_arrow(x_mid, base_y + block_h + 1.73, x_mid, base_y + block_h + 1.95)
        draw_arrow(x_mid, base_y + block_h + 2.53, x_mid, base_y + block_h + 2.95)

        ax.text(x_mid, base_y + block_h + 3.05, "Output Probabilities", fontsize=13, ha="center")

        ax.text(
            x_mid,
            9.6,
            "BERT/RoBERTa\n",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.set_xlim(0.8, 6.4)
        ax.set_ylim(-0.4, 9.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def train(epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5, weight_decay: float = 0.01, max_length: int = 256):
        df = get_datasets()

        X_texts = df["Text"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        X_train, X_eval, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

        model = BertModel.from_pretrained(
            MODEL_NAME,
            label_map=label_map
        )

        tokenized_train = model.tokenizer(X_train, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        tokenized_eval = model.tokenizer(X_eval, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

        train_dataset = TextClassificationDataset(tokenized_train, y_train)
        eval_dataset = TextClassificationDataset(tokenized_eval, y_eval)

        training_args = TrainingArguments(
            output_dir=main_folder / "results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_strategy="epoch",
            logging_steps=10,
            dataloader_num_workers=4 if torch_utils.device.type == "cpu" else 0,
            dataloader_pin_memory=torch_utils.device.type == "cuda",
            gradient_accumulation_steps=2,
            report_to="none",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_logits,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

        torch.save(
            {
                "model_name": MODEL_NAME,
                "model_state": model.model.state_dict(),
                "tokenizer": model.tokenizer,
                "label_map": label_map,
            },
            main_folder / "bert.pt",
        )

        print("Model saved to", main_folder / "bert.pt")

    @staticmethod
    def plot_architecture(max_length: int = 256):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # roberta-base defaults used in this trainer
        hidden_size = 768
        n_layers = 12
        n_heads = 12

        fig, ax = plt.subplots(figsize=(7.6, 9.0))
        ax.axis("off")

        def box(x, y, w, h, text, color, fontsize=10):
            rect = mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="black",
                linewidth=1.7,
                alpha=0.92,
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

        def arrow(x1, y1, x2, y2, lw=1.8):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color="black"),
            )

        x = 3.8
        w = 2.8
        h = 0.62

        box(x - w / 2, 0.25, w, h, f"Input IDs + Attention Mask\n[B, T], T <= {max_length}", "#D9E1F2", 10)
        box(x - w / 2, 1.20, w, h, "Token/Position Embeddings", "#F8D7DA", 10)
        box(x - w / 2, 2.20, w, 3.45, f"BERT Encoder Stack\nN×={n_layers}, heads={n_heads}, hidden={hidden_size}", "#F4F6F7", 11)

        # inside encoder block (paper-ish style)
        inner_w = 2.25
        inner_x = x - inner_w / 2
        box(inner_x, 2.55, inner_w, 0.55, "Multi-Head Self-Attention", "#FDE0B6", 9)
        box(inner_x + 0.15, 3.23, inner_w - 0.30, 0.40, "Add & Norm", "#F7F1B5", 9)
        box(inner_x, 3.80, inner_w, 0.55, "Feed Forward", "#B9DDF1", 9)
        box(inner_x + 0.15, 4.48, inner_w - 0.30, 0.40, "Add & Norm", "#F7F1B5", 9)

        box(x - w / 2, 6.05, w, h, "[CLS] / Sequence Pooling", "#D9D9D9", 10)
        box(x - w / 2, 7.00, w, h, "Dropout + Linear Classifier", "#CFE2F3", 10)
        box(x - w / 2, 7.95, w, h, "Softmax -> Class Probabilities", "#D9EAD3", 10)

        # vertical flow
        arrow(x, 0.87, x, 1.20)
        arrow(x, 1.82, x, 2.20)
        arrow(x, 5.65, x, 6.05)
        arrow(x, 6.67, x, 7.00)
        arrow(x, 7.62, x, 7.95)
        arrow(x, 8.57, x, 8.95)

        ax.text(x, 9.15, "BERT/RoBERTa Sequence Classification Architecture", ha="center", fontsize=12, fontweight="bold")
        ax.set_xlim(1.2, 6.4)
        ax.set_ylim(0.0, 9.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pipeline():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        col1 = [
            ("", "", "#4C72B0"),
            ("Load Dataset", "get_datasets()", "#4C72B0"),
            ("Label Encoding", "string -> id", "#8C8C8C"),
            ("Train/Val Split", "stratified", "#8C8C8C"),
            ("Load Tokenizer", f"{MODEL_NAME}", "#4C72B0"),
            ("Tokenize", "padding + truncation", "#DD8452"),
            ("Build Datasets", "TextClassificationDataset", "#8C8C8C"),
        ]

        col2 = [
            ("", "", "#55A868"),
            ("Load Pretrained", "AutoModelForSequenceClassification", "#55A868"),
            ("Trainer Setup", "TrainingArguments + Trainer", "#55A868"),
            ("Fine-tuning", "trainer.train()", "#55A868"),
            ("Validation", "trainer.evaluate()", "#55A868"),
            ("Best Model", "load_best_model_at_end", "#C44E52"),
            ("Save Checkpoint", "bert.pt", "#937860"),
        ]

        box_w, box_h, gap = 2.8, 0.70, 0.38
        col_sep = 0.7
        n_rows = max(len(col1), len(col2))
        total_h = n_rows * (box_h + gap) - gap

        fig, ax = plt.subplots(figsize=(9.0, total_h + 1.0))
        ax.axis("off")

        def draw_column(steps, x_offset, title):
            ax.text(
                x_offset + box_w / 2,
                total_h + box_h * 0.72,
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
                    y + box_h / 2 + 0.09,
                    name,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white",
                )
                ax.text(
                    x_offset + box_w / 2,
                    y + box_h / 2 - 0.13,
                    detail,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white",
                )

        draw_column(col1, 0.0, "1) Data Preparation")
        draw_column(col2, box_w + col_sep, "2) Fine-Tuning Loop")

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
        ax.set_title("BERT - Training Pipeline", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    BertTrainer.train()
