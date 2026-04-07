import copy
import numpy as np
import scipy.sparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich import box

from models.dnn.model import NumpyModel
from models.dnn.dnnutils import train_test_split, TFIDF
from models.dnn.dnnutils.losses import CategoricalCrossEntropy
from prepare.dataset import get_datasets
from utils.dataset import preprocess_text, normalize_text, extract_features
from utils.model import main_folder

SEED = 42
np.random.seed(SEED)


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best = None
        self.counter = 0
        self.best_state = None

    def step(self, metric, model):
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
            self.best_state = copy.deepcopy(model.nn.layers)
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.nn.layers = self.best_state


class TrainNumpy:
    @staticmethod
    def _compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = (len(y) / (n_classes * counts)).astype(np.float32)
        return (weights / np.mean(weights)).astype(np.float32)

    @staticmethod
    def _balanced_epoch_indices(y: np.ndarray, batch_size: int, n_classes: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        class_indices = [np.where(y == cls)[0] for cls in range(n_classes)]
        per_class = max(1, batch_size // n_classes)
        n_batches = int(np.ceil(len(y) / batch_size))

        indices = []
        all_idx = np.arange(len(y))
        for _ in range(n_batches):
            batch = []
            for cls_idx in class_indices:
                if len(cls_idx) == 0:
                    continue
                replace = len(cls_idx) < per_class
                batch.extend(rng.choice(cls_idx, size=per_class, replace=replace).tolist())

            missing = batch_size - len(batch)
            if missing > 0:
                batch.extend(rng.choice(all_idx, size=missing, replace=len(all_idx) < missing).tolist())

            rng.shuffle(batch)
            indices.extend(batch[:batch_size])

        return np.asarray(indices[: n_batches * batch_size], dtype=np.int64)

    @staticmethod
    def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
        accs = np.zeros(n_classes, dtype=np.float32)
        for cls in range(n_classes):
            cls_mask = y_true == cls
            if np.any(cls_mask):
                accs[cls] = np.mean(y_pred[cls_mask] == y_true[cls_mask])
        return accs

    @staticmethod
    def _predict_batches(nn, X, batch_size: int = 512) -> np.ndarray:
        """Forward pass in batches so we never densify the full sparse matrix at once."""
        parts = []
        for start in range(0, X.shape[0], batch_size):
            chunk = X[start : start + batch_size]
            if scipy.sparse.issparse(chunk):
                chunk = chunk.toarray()
            parts.append(nn.forward_propagation(chunk, training=False))
        return np.vstack(parts)

    @staticmethod
    def build_tfidf_vectorizer(texts: list[str], ngram_range=(1, 2), max_features=5000) -> TFIDF:
        tfidf = TFIDF(ngram_range=ngram_range, max_features=max_features)
        tfidf.fit(texts)
        return tfidf

    @staticmethod
    def tfidf_char(texts: list[str], ngram_range=(2, 5), max_features=5000) -> TFIDF:
        tfidf = TFIDF(analyzer="char", ngram_range=ngram_range, max_features=max_features)
        tfidf.fit(texts)
        return tfidf

    @staticmethod
    def train(
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.005,
        use_class_weights: bool = True,
        use_balanced_sampling: bool = False,
        hidden_units: tuple[int, int, int] = (512, 256, 128),
        dropout_rates: tuple[float, float, float] = (0.2, 0.1, 0.1),
        enable_tfidf_char: bool = True,
        enable_handcrafted: bool = False,
        tfidf_word_features: int = 10000,
        tfidf_char_features: int = 2000,
    ):
        if use_class_weights and use_balanced_sampling:
            print(
                "Both class weighting and balanced sampling were requested; "
                "disabling balanced sampling to avoid over-correcting minority classes."
            )
            use_balanced_sampling = False

        df = get_datasets(submission_round=3, balance=True, target_per_class=5000)

        df["Text_clean"] = df["Text"].apply(preprocess_text)
        X_word_texts = df["Text_clean"].tolist()
        X_char_texts = None
        if enable_tfidf_char:
            df["Text_norm"] = df["Text"].apply(normalize_text)
            X_char_texts = df["Text_norm"].tolist()

        y_labels = df["Label"].tolist()
        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_labels = [label_map[label] for label in y_labels]

        X_train_word_texts, X_eval_word_texts, y_train, y_eval = train_test_split(
            X_word_texts,
            y_labels,
            test_size=0.2,
            random_state=42,
        )

        tfidf_word = TrainNumpy.build_tfidf_vectorizer(X_train_word_texts, ngram_range=(1, 2), max_features=tfidf_word_features)

        X_train_word = tfidf_word.transform(X_train_word_texts)
        X_eval_word = tfidf_word.transform(X_eval_word_texts)

        X_train = X_train_word
        X_eval = X_eval_word
        tfidf_char = None
        hand_feature_names = None
        hand_mean = None
        hand_std = None

        if enable_tfidf_char:
            X_train_char_texts, X_eval_char_texts, _, _ = train_test_split(
                X_char_texts,
                y_labels,
                test_size=0.2,
                random_state=42,
            )

            tfidf_char = TrainNumpy.tfidf_char(
                X_train_char_texts,
                ngram_range=(2, 5),
                max_features=tfidf_char_features,
            )
            X_train_char = tfidf_char.transform(X_train_char_texts)
            X_eval_char = tfidf_char.transform(X_eval_char_texts)

            X_train = scipy.sparse.hstack([X_train, X_train_char], format="csr").astype(np.float32)
            X_eval = scipy.sparse.hstack([X_eval, X_eval_char], format="csr").astype(np.float32)

        if enable_handcrafted:
            df["Features"] = df["Text"].apply(lambda txt: extract_features(txt, preprocess_text(txt)))
            hand_feature_names = list(df["Features"].iloc[0].keys())
            x_hand_all = [list(feat[name] for name in hand_feature_names) for feat in df["Features"]]

            x_hand_train, x_hand_eval, _, _ = train_test_split(x_hand_all, y_labels, test_size=0.2, random_state=42)

            x_hand_train = np.array(x_hand_train, dtype=np.float32)
            x_hand_eval = np.array(x_hand_eval, dtype=np.float32)
            hand_mean = np.mean(x_hand_train, axis=0)
            hand_std = np.std(x_hand_train, axis=0) + 1e-8
            x_hand_train = (x_hand_train - hand_mean) / hand_std
            x_hand_eval = (x_hand_eval - hand_mean) / hand_std

            X_train = scipy.sparse.hstack([X_train, scipy.sparse.csr_matrix(x_hand_train)], format="csr").astype(np.float32)
            X_eval = scipy.sparse.hstack([X_eval, scipy.sparse.csr_matrix(x_hand_eval)], format="csr").astype(np.float32)

        y_train = np.asarray(y_train, dtype=np.int64)
        y_eval = np.asarray(y_eval, dtype=np.int64)

        y_train_one_hot = np.eye(len(unique_labels), dtype=np.float32)[y_train]

        class_weights = TrainNumpy._compute_class_weights(y_train, len(unique_labels))
        train_counts = np.bincount(y_train, minlength=len(unique_labels))
        print("Class counts (train):", train_counts.tolist())
        print("Class weights:", class_weights.round(4).tolist())
        print(
            "Sampling strategy:",
            "balanced" if use_balanced_sampling else "random",
            "| Weighting:",
            "enabled" if use_class_weights else "disabled",
        )
        print("TF-IDF strategy: word=cleaned text | char=normalized raw text")

        model = NumpyModel.create(
            input_dim=X_train.shape[1],
            label_map=label_map,
            tfid_word=tfidf_word,
            tfid_char=tfidf_char,
            hand_feature_names=hand_feature_names,
            hand_mean=hand_mean,
            hand_std=hand_std,
            hidden_units=hidden_units,
            dropout_rates=dropout_rates,
            loss=CategoricalCrossEntropy(),
        )

        early_stop = EarlyStopping(patience=5)
        history = {"loss": [], "train_acc": [], "val_acc": []}

        console = Console()
        progress = Progress(
            TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[epochs_total]}"),
            BarColumn(bar_width=30),
            TextColumn("loss=[yellow]{task.fields[loss]:.4f}[/] train=[green]{task.fields[train_acc]:.4f}[/] val=[magenta]{task.fields[val_acc]:.4f}[/]"),
            TimeElapsedColumn(),
            console=console,
        )
        epoch_task = progress.add_task(
            "training",
            total=epochs,
            epoch=0,
            epochs_total=epochs,
            loss=0.0,
            train_acc=0.0,
            val_acc=0.0,
        )

        with Live(progress, console=console, refresh_per_second=4):
            for epoch in range(epochs):
                # mini-batch training
                if use_balanced_sampling:
                    indices = TrainNumpy._balanced_epoch_indices(
                        y_train,
                        batch_size=batch_size,
                        n_classes=len(unique_labels),
                        seed=SEED + epoch,
                    )
                else:
                    indices = np.random.permutation(X_train.shape[0])

                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, X_train.shape[0], batch_size):
                    batch_idx = indices[start:start + batch_size]
                    X_batch = X_train[batch_idx]
                    if scipy.sparse.issparse(X_batch):
                        X_batch = X_batch.toarray()
                    y_batch_one_hot = y_train_one_hot[batch_idx]
                    y_batch = y_train[batch_idx]

                    probs = model.nn.forward_propagation(X_batch, training=True)
                    error = model.loss.derivative(y_batch_one_hot, probs)

                    if use_class_weights:
                        sample_weights = class_weights[y_batch].reshape(-1, 1)
                        error = error * sample_weights
                        error = error / np.mean(sample_weights)
                        sample_losses = -np.sum(y_batch_one_hot * np.log(np.clip(probs, 1e-15, 1 - 1e-15)), axis=1)
                        epoch_loss += np.mean(sample_losses * sample_weights.reshape(-1))
                    else:
                        epoch_loss += model.loss.loss(y_batch_one_hot, probs)

                    model.nn.backward_propagation(error, learning_rate=learning_rate)
                    n_batches += 1

                avg_loss = epoch_loss / n_batches

                train_probs = TrainNumpy._predict_batches(model.nn, X_train)
                train_preds = np.argmax(train_probs, axis=1)
                train_acc = np.mean(train_preds == y_train)

                eval_probs = TrainNumpy._predict_batches(model.nn, X_eval)
                eval_preds = np.argmax(eval_probs, axis=1)
                eval_acc = np.mean(eval_preds == y_eval)
                per_class_eval = TrainNumpy._per_class_accuracy(y_eval, eval_preds, len(unique_labels))

                history["loss"].append(avg_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(eval_acc)

                progress.update(
                    epoch_task,
                    advance=1,
                    epoch=epoch + 1,
                    loss=avg_loss,
                    train_acc=train_acc,
                    val_acc=eval_acc,
                )

                if early_stop.step(eval_acc, model):
                    console.print(f"[bold yellow]Early stopping[/] at epoch {epoch + 1}. Best val_acc={early_stop.best:.4f}")
                    break

        early_stop.restore(model)

        # print per-class summary after training
        console = Console()
        eval_probs = TrainNumpy._predict_batches(model.nn, X_eval)
        eval_preds = np.argmax(eval_probs, axis=1)
        eval_acc = np.mean(eval_preds == y_eval)
        t = Table(title="Final Results", box=box.ROUNDED)
        t.add_column("Class", style="cyan")
        t.add_column("Val Accuracy", justify="right")
        final_per_class = TrainNumpy._per_class_accuracy(y_eval, eval_preds, len(unique_labels))
        for label, acc in zip(unique_labels, final_per_class):
            t.add_row(label, f"{acc:.4f}")
        t.add_section()
        t.add_row("[bold]Overall[/]", f"[bold]{eval_acc:.4f}[/]")
        console.print(t)

        model.save(main_folder / "numpy-dnn.pkl.gz")
        return history

    @staticmethod
    def plot_history(history: dict):
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, history["loss"], color="#C44E52", linewidth=2)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["train_acc"], label="train", color="#4C72B0", linewidth=2)
        ax2.plot(epochs, history["val_acc"], label="val", color="#55A868", linewidth=2, linestyle="--")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("NumPy DNN — Training History", fontweight="bold")
        plt.tight_layout()
        output_dir = main_folder / ".." / "docs" / "article" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "numpy_dnn_train.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print("History plot saved to", output_path)
        plt.show()

    @staticmethod
    def plot_pipeline(
        enable_tfidf_char: bool = True,
        enable_handcrafted: bool = False,
    ):
        """Draws a two-column flowchart of the training pipeline."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # ── column 1: data preparation ─────────────────────────────────────
        col1 = [
            ("",                "",                           "#4C72B0"),
            ("Load Dataset",    "get_datasets()",             "#4C72B0"),
            ("Preprocess Text", "clean + normalize",          "#4C72B0"),
            ("Train/Val Split", "80% / 20%  stratified",      "#8C8C8C"),
            ("Word TF-IDF",     "1-2 grams  max 10k",         "#DD8452"),
        ]
        if enable_tfidf_char:
            col1.append(("Char TF-IDF",  "2-5 grams  max 2k",  "#DD8452"))
        if enable_handcrafted:
            col1.append(("Handcrafted",  "stylistic features",  "#DD8452"))
        col1.append(("Concat Features", "hstack all blocks",    "#8C8C8C"))

        # ── column 2: training loop ────────────────────────────────────────
        col2 = [
            ("",                 "",                           "#55A868"),
            ("Build Model",      "Dense→ReLU→Dropout ×3",      "#55A868"),
            ("Mini-batch Train", "forward → loss → backward",  "#55A868"),
            ("Class Weights",    "weighted error per sample",  "#55A868"),
            ("Eval per Epoch",   "val_acc + per-class acc",    "#55A868"),
            ("Early Stopping",   "patience=5  restore best",   "#C44E52"),
            ("Save Model",       "numpy-dnn.pkl.gz",           "#937860"),
        ]

        BOX_W, BOX_H, GAP = 2.4, 0.68, 0.42
        COL_SEP = 0.5   # horizontal gap between columns
        n_rows = max(len(col1), len(col2))
        total_h = n_rows * (BOX_H + GAP) - GAP

        fig, ax = plt.subplots(figsize=(7.5, total_h + 1.0))
        ax.axis("off")

        def draw_col(steps, x_off, label):
            ax.text(x_off + BOX_W / 2, total_h + BOX_H * 0.6, label,
                    ha="center", va="center", fontsize=10, fontweight="bold", color="#333")
            for i, (title, detail, color) in enumerate(steps):
                y = total_h - i * (BOX_H + GAP)
                if i > 0:
                    ax.annotate(
                        "", xy=(x_off + BOX_W / 2, y + BOX_H),
                        xytext=(x_off + BOX_W / 2, y + BOX_H + GAP),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
                    )
                rect = mpatches.FancyBboxPatch(
                    (x_off, y), BOX_W, BOX_H,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.88,
                )
                ax.add_patch(rect)
                ax.text(x_off + BOX_W / 2, y + BOX_H / 2 + 0.10, title,
                        ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
                ax.text(x_off + BOX_W / 2, y + BOX_H / 2 - 0.15, detail,
                        ha="center", va="center", fontsize=7.5, color="white")

        draw_col(col1, x_off=0, label="① Data Preparation")
        draw_col(col2, x_off=BOX_W + COL_SEP, label="② Training Loop")

        # arrow connecting last box of col1 to first box of col2
        y_last_col1 = total_h - (len(col1) - 1) * (BOX_H + GAP)
        y_first_col2 = total_h
        mid_y = (y_last_col1 + y_first_col2) / 2
        x_mid = BOX_W + COL_SEP / 2
        ax.annotate(
            "", xy=(BOX_W + COL_SEP, y_first_col2 + BOX_H / 2),
            xytext=(BOX_W, y_last_col1 + BOX_H / 2),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2,
                            connectionstyle="arc3,rad=-0.3"),
        )

        ax.set_xlim(-0.2, 2 * BOX_W + COL_SEP + 0.2)
        ax.set_ylim(-0.4, total_h + BOX_H + 0.8)
        ax.set_title("NumPy DNN — Training Pipeline", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    history = TrainNumpy.train()
    TrainNumpy.plot_history(history)
