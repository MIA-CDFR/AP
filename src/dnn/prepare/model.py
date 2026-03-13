import gzip
import pickle
from collections.abc import Iterable

import pandas as pd
import numpy as np

from dnn.prepare.dataset import DatasetLoader
from dnn.prepare.feature import build_text_vector
from dnn.nn import NeuralNetwork
from dnn.layers import DenseLayer
from dnn.layers.activation import ReLU, Softmax, Dropout


class Model:
    def __init__(self, n_classes: int = 6):
        self.nn = NeuralNetwork()
        self.nn.add_layer(DenseLayer(256))
        self.nn.add_layer(ReLU())
        self.nn.add_layer(Dropout(0.3))
        self.nn.add_layer(DenseLayer(128))
        self.nn.add_layer(ReLU())
        self.nn.add_layer(Dropout(0.2))
        self.nn.add_layer(DenseLayer(n_classes))
        self.nn.add_layer(Softmax())

        self.tfidf_word = None
        self.tfidf_char = None
        self.hand_mean = None
        self.hand_std = None
        self.hand_feature_names = None
        self.class_names = None

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 10, batch_size: int = 32):
        history = self.nn.fit(
            X_train,
            y_train,
            epochs=epochs,
            learning_rate=0.01,
            batch_size=batch_size,
            x_val=X_val,
            y_val=y_val,
            verbose_every=25,
            patience=30,
            min_delta=1e-4,
            lr_decay=0.5,
            lr_patience=10,
        )
        print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Best Val Loss: {min(history['val_loss']):.4f}")
        return history

    def _has_preprocessors(self) -> bool:
        return all(
            item is not None
            for item in [
                self.tfidf_word,
                self.tfidf_char,
                self.hand_mean,
                self.hand_std,
                self.hand_feature_names,
                self.class_names,
            ]
        )

    def _vectorize_text_batch(self, texts: Iterable[str]) -> np.ndarray:
        if not self._has_preprocessors():
            raise ValueError("Preprocessors not attached. Call attach_preprocessors(datasets) before text prediction.")

        vectors = [
            build_text_vector(
                text,
                self.tfidf_word,
                self.tfidf_char,
                self.hand_mean,
                self.hand_std,
                self.hand_feature_names,
            )[0]
            for text in texts
        ]
        return np.array(vectors, dtype=np.float32)

    def _labels_to_indices(self, labels) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.size == 0:
            return labels_array.astype(int)

        if np.issubdtype(labels_array.dtype, np.integer):
            return labels_array.astype(int)

        if self.class_names is None:
            raise ValueError("class_names must be attached to convert string labels into indices.")

        label_to_num = {label: i for i, label in enumerate(self.class_names)}
        return np.array([label_to_num[label] for label in labels_array], dtype=int)

    def predict(self, X, y=None):
        if isinstance(X, np.ndarray):
            x_input = X
        elif isinstance(X, pd.Series):
            x_input = self._vectorize_text_batch(X.astype(str).tolist())
        elif isinstance(X, list) and (len(X) == 0 or isinstance(X[0], str)):
            x_input = self._vectorize_text_batch(X)
        else:
            raise TypeError("X must be a numpy matrix of features or a sequence/Series of raw texts.")

        predictions = self.nn.predict(x_input)
        predicted_labels = np.argmax(predictions, axis=1)

        num_to_label = {i: label for i, label in enumerate(self.class_names)} if self.class_names is not None else None
        y_num = self._labels_to_indices(y) if y is not None else None

        if y is not None:
            accuracy = np.mean(predicted_labels == y_num)
            if self.class_names is not None:
                n_classes = len(self.class_names)
                print("Predicted label distribution:", np.bincount(predicted_labels, minlength=n_classes))
                print("True label distribution:", np.bincount(y_num, minlength=n_classes))
            print(f"Test Accuracy: {accuracy:.4f}")

        if num_to_label is None:
            return y, predicted_labels.tolist()

        self.print_confusion_matrix(y, [num_to_label[i] for i in predicted_labels])

        return [num_to_label[i] for i in predicted_labels]

    def confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        y_true_indices = self._labels_to_indices(y_true)
        y_pred_indices = self._labels_to_indices(y_pred)

        n_classes = len(self.class_names) if self.class_names is not None else max(
            y_true_indices.max(initial=0), y_pred_indices.max(initial=0)
        ) + 1

        matrix = np.zeros((n_classes, n_classes), dtype=int)
        for true_label, pred_label in zip(y_true_indices, y_pred_indices):
            matrix[true_label, pred_label] += 1
        return matrix

    def print_confusion_matrix(self, y_true, y_pred):
        matrix = self.confusion_matrix(y_true, y_pred)
        print("Confusion Matrix (rows=true, cols=pred):")
        print(matrix)
        if self.class_names is not None:
            print("Classes:", self.class_names)
        return matrix

    def attach_preprocessors(self, datasets: DatasetLoader):
        self.tfidf_word = datasets.tfidf_word
        self.tfidf_char = datasets.tfidf_char
        self.hand_mean = datasets.hand_mean
        self.hand_std = datasets.hand_std
        self.hand_feature_names = datasets.hand_feature_names
        self.class_names = datasets.class_names

    def predict_text(self, text: str) -> str:
        if not self._has_preprocessors():
            raise ValueError("Preprocessors not attached. Call attach_preprocessors(datasets) before predict_text.")

        x = build_text_vector(
            text,
            self.tfidf_word,
            self.tfidf_char,
            self.hand_mean,
            self.hand_std,
            self.hand_feature_names,
        )
        probs = self.nn.predict(x)[0]
        pred_idx = int(np.argmax(probs))
        return self.class_names[pred_idx]

    def save(self, path: str):
        if self.tfidf_word is not None and hasattr(self.tfidf_word, "vocab"):
            self.tfidf_word.vocab.frequencies.clear()
        if self.tfidf_char is not None and hasattr(self.tfidf_char, "vocab"):
            self.tfidf_char.vocab.frequencies.clear()

        with gzip.open(path, "wb", compresslevel=9) as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Model":
        try:
            with gzip.open(path, "rb") as file:
                return pickle.load(file)
        except (OSError, EOFError):
            with open(path, "rb") as file:
                return pickle.load(file)


class LinearRegressionModel(Model):
    def __init__(self, n_classes: int = 6, l2: float = 1e-4):
        super().__init__(n_classes=n_classes)
        self.l2 = l2
        self.linear_weights = None
        self.linear_bias = None

    def _scores(self, x_input: np.ndarray) -> np.ndarray:
        if self.linear_weights is None or self.linear_bias is None:
            raise ValueError("LinearRegressionModel is not trained. Call train(...) first.")
        return x_input @ self.linear_weights + self.linear_bias

    def _prepare_targets(self, y):
        y_array = np.asarray(y)

        if np.issubdtype(y_array.dtype, np.integer):
            y_num = y_array.astype(int)
            if self.class_names is None:
                self.class_names = [str(i) for i in sorted(np.unique(y_num))]
        else:
            if self.class_names is None:
                self.class_names = sorted(np.unique(y_array).tolist())
            y_num = self._labels_to_indices(y_array)

        return y_num

    def train(self, X_train, y_train, X_val=None, y_val=None):

        x_train = np.asarray(X_train, dtype=np.float32)
        y_train_num = self._prepare_targets(y_train)

        n_classes = len(self.class_names)
        y_train_one_hot = np.eye(n_classes, dtype=np.float32)[y_train_num]

        x_bias = np.hstack([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)])
        reg = np.eye(x_bias.shape[1], dtype=np.float32) * np.float32(self.l2)
        reg[-1, -1] = 0.0

        lhs = x_bias.T @ x_bias + reg
        rhs = x_bias.T @ y_train_one_hot
        beta = np.linalg.solve(lhs, rhs)

        self.linear_weights = beta[:-1, :]
        self.linear_bias = beta[-1, :]

        train_scores = self._scores(x_train)
        train_loss = float(np.mean((train_scores - y_train_one_hot) ** 2))
        train_acc = float(np.mean(np.argmax(train_scores, axis=1) == y_train_num))

        history = {
            "train_loss": [train_loss],
            "train_acc": [train_acc],
            "val_loss": [],
            "val_acc": [],
        }

        print(f"Final Train Loss: {train_loss:.4f}")
        print(f"Final Train Accuracy: {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            x_val = np.asarray(X_val, dtype=np.float32)
            y_val_num = self._prepare_targets(y_val)
            y_val_one_hot = np.eye(n_classes, dtype=np.float32)[y_val_num]
            val_scores = self._scores(x_val)
            val_loss = float(np.mean((val_scores - y_val_one_hot) ** 2))
            val_acc = float(np.mean(np.argmax(val_scores, axis=1) == y_val_num))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")

        return history

    def predict(self, X, y=None):
        if isinstance(X, np.ndarray):
            x_input = X.astype(np.float32)
        elif isinstance(X, pd.Series):
            x_input = self._vectorize_text_batch(X.astype(str).tolist())
        elif isinstance(X, list) and (len(X) == 0 or isinstance(X[0], str)):
            x_input = self._vectorize_text_batch(X)
        else:
            raise TypeError("X must be a numpy matrix of features or a sequence/Series of raw texts.")

        scores = self._scores(x_input)
        predicted_labels_num = np.argmax(scores, axis=1)

        num_to_label = {i: label for i, label in enumerate(self.class_names)} if self.class_names is not None else None

        if y is not None:
            y_num = self._prepare_targets(y)
            accuracy = np.mean(predicted_labels_num == y_num)
            if self.class_names is not None:
                n_classes = len(self.class_names)
                print("Predicted label distribution:", np.bincount(predicted_labels_num, minlength=n_classes))
                print("True label distribution:", np.bincount(y_num, minlength=n_classes))
            print(f"Test Accuracy: {accuracy:.4f}")

        if num_to_label is None:
            return predicted_labels_num.tolist()

        predicted_labels = [num_to_label[i] for i in predicted_labels_num]
        if y is not None:
            self.print_confusion_matrix(y, predicted_labels)
        return predicted_labels

    def predict_text(self, text: str) -> str:
        if not self._has_preprocessors():
            raise ValueError("Preprocessors not attached. Call attach_preprocessors(datasets) before predict_text.")

        x = build_text_vector(
            text,
            self.tfidf_word,
            self.tfidf_char,
            self.hand_mean,
            self.hand_std,
            self.hand_feature_names,
        ).astype(np.float32)
        scores = self._scores(x)
        pred_idx = int(np.argmax(scores))
        if self.class_names is None:
            return str(pred_idx)
        return self.class_names[pred_idx]
        