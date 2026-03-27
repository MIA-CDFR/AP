import copy
import numpy as np

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
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.005,
        use_class_weights: bool = True,
        use_balanced_sampling: bool = False,
        hidden_units: tuple[int, int] = (1024, 512),
        dropout_rates: tuple[float, float] = (0.2, 0.1),
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

        df = get_datasets()

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

            X_train = np.hstack([X_train, X_train_char]).astype(np.float32, copy=False)
            X_eval = np.hstack([X_eval, X_eval_char]).astype(np.float32, copy=False)

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

            X_train = np.hstack([X_train, x_hand_train]).astype(np.float32, copy=False)
            X_eval = np.hstack([X_eval, x_hand_eval]).astype(np.float32, copy=False)

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
                indices = np.random.permutation(len(X_train))

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X_train[batch_idx]
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

            train_probs = model.nn.forward_propagation(X_train, training=False)
            train_preds = np.argmax(train_probs, axis=1)
            train_acc = np.mean(train_preds == y_train)

            eval_probs = model.nn.forward_propagation(X_eval, training=False)
            eval_preds = np.argmax(eval_probs, axis=1)
            eval_acc = np.mean(eval_preds == y_eval)
            per_class_eval = TrainNumpy._per_class_accuracy(y_eval, eval_preds, len(unique_labels))

            print(
                f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={eval_acc:.4f}"
            )
            per_class_msg = " | ".join(
                f"{label}: {acc:.3f}" for label, acc in zip(unique_labels, per_class_eval)
            )
            print(f"  Val per-class acc -> {per_class_msg}")

            if early_stop.step(eval_acc, model):
                print(f"Early stopping at epoch {epoch + 1}. Best val_acc={early_stop.best:.4f}")
                break

        early_stop.restore(model)
        eval_probs = model.nn.forward_propagation(X_eval, training=False)
        eval_preds = np.argmax(eval_probs, axis=1)
        eval_acc = np.mean(eval_preds == y_eval)
        print(f"\nFinal validation accuracy (best model): {eval_acc:.4f}")

        model.save(main_folder / "numpy-dnn.pkl.gz")

if __name__ == "__main__":
    TrainNumpy.train()
