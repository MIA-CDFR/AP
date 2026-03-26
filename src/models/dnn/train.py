import copy
import numpy as np

from models.dnn.model import NumpyModel
from models.dnn.dnnutils import train_test_split, TFIDF
from models.dnn.dnnutils.losses import CategoricalCrossEntropy
from prepare.dataset import get_datasets
from utils.dataset import preprocess_text, extract_features
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
    def train(epochs: int = 50, batch_size: int = 64, learning_rate: float = 0.005):
        df = get_datasets()

        df["Text_clean"] = df["Text"].apply(preprocess_text)
        X_texts = df["Text_clean"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        df["Features"] = df["Text"].apply(lambda txt: extract_features(txt, preprocess_text(txt)))
        hand_feature_names = list(df["Features"].iloc[0].keys())
        x_hand_all = [list(feat[name] for name in hand_feature_names) for feat in df["Features"]]

        X_train_texts, X_eval_texts, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42)
        x_hand_train, x_hand_eval, _, _ = train_test_split(x_hand_all, y_labels, test_size=0.2, random_state=42)

        tfidf_word = TFIDF(ngram_range=(1, 2), max_features=10000)
        tfidf_char = TFIDF(ngram_range=(2, 5), max_features=10000)

        X_train_word = tfidf_word.fit_transform(X_train_texts)
        X_train_char = tfidf_char.fit_transform(X_train_texts)

        X_eval_word = tfidf_word.transform(X_eval_texts)
        X_eval_char = tfidf_char.transform(X_eval_texts)

        x_hand_train = np.array(x_hand_train, dtype=np.float32)
        x_hand_eval = np.array(x_hand_eval, dtype=np.float32)
        hand_mean = np.mean(x_hand_train, axis=0)
        hand_std = np.std(x_hand_train, axis=0) + 1e-8
        x_hand_train = (x_hand_train - hand_mean) / hand_std
        x_hand_eval = (x_hand_eval - hand_mean) / hand_std

        X_train = np.hstack([X_train_word, X_train_char, x_hand_train]).astype(np.float32)
        X_eval = np.hstack([X_eval_word, X_eval_char, x_hand_eval]).astype(np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        y_eval = np.asarray(y_eval, dtype=np.int64)
        y_train_one_hot = np.eye(len(unique_labels), dtype=np.float32)[y_train]

        model = NumpyModel.create(
            input_dim=X_train.shape[1],
            tfid_word=tfidf_word,
            tfid_char=tfidf_char,
            label_map=label_map,
            hand_feature_names=hand_feature_names,
            hand_mean=hand_mean,
            hand_std=hand_std,
            loss=CategoricalCrossEntropy(),
        )

        early_stop = EarlyStopping(patience=5)

        for epoch in range(epochs):
            # mini-batch training
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X_train[batch_idx]
                y_batch_one_hot = y_train_one_hot[batch_idx]

                probs = model.nn.forward_propagation(X_batch, training=True)
                epoch_loss += model.loss.loss(y_batch_one_hot, probs)
                error = model.loss.derivative(y_batch_one_hot, probs)
                model.nn.backward_propagation(error, learning_rate=learning_rate)
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            train_probs = model.nn.forward_propagation(X_train, training=False)
            train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train)

            eval_probs = model.nn.forward_propagation(X_eval, training=False)
            eval_acc = np.mean(np.argmax(eval_probs, axis=1) == y_eval)

            print(f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | train_acc={train_acc:.4f} | val_acc={eval_acc:.4f}")

            if early_stop.step(eval_acc, model):
                print(f"Early stopping at epoch {epoch + 1}. Best val_acc={early_stop.best:.4f}")
                break

        early_stop.restore(model)
        eval_probs = model.nn.forward_propagation(X_eval, training=False)
        eval_acc = np.mean(np.argmax(eval_probs, axis=1) == y_eval)
        print(f"\nFinal validation accuracy (best model): {eval_acc:.4f}")

        model.save(main_folder / "numpy-dnn.pkl.gz")

if __name__ == "__main__":
    TrainNumpy.train()
