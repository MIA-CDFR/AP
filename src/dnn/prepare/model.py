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
        self.nn.add_layer(DenseLayer(512))
        self.nn.add_layer(ReLU())
        self.nn.add_layer(Dropout(0.3))
        self.nn.add_layer(DenseLayer(512))
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

