import pickle
import gzip
import numpy as np

from models.dnn.classifiers import NeuralNetwork
from models.dnn.classifiers.layers import DenseLayer, ReLU, Dropout, Softmax

from models.dnn.dnnutils.losses import LossFunction
from utils.dataset import preprocess_text, extract_features


class NumpyModel:
    def __init__(
        self,
        n_classes: int = 5,
        hidden_units: tuple[int, int] = (512, 256),
        dropout_rates: tuple[float, float] = (0.3, 0.2),
    ):
        if len(hidden_units) != 2:
            raise ValueError("hidden_units must contain exactly two layer sizes")
        if len(dropout_rates) != 2:
            raise ValueError("dropout_rates must contain exactly two rates")

        self.nn = NeuralNetwork()
        self.nn.add_layer(DenseLayer(hidden_units[0]))
        self.nn.add_layer(ReLU())
        self.nn.add_layer(Dropout(dropout_rates[0]))
        self.nn.add_layer(DenseLayer(hidden_units[1]))
        self.nn.add_layer(ReLU())
        self.nn.add_layer(Dropout(dropout_rates[1]))
        self.nn.add_layer(DenseLayer(n_classes))
        self.nn.add_layer(Softmax())

        self.tfidf_word = None
        self.tfidf_char = None
        self.label_map = None
        self.inverse_label_map = None
        self.hand_feature_names = None
        self.hand_mean = None
        self.hand_std = None

        self.loss = None

    @classmethod
    def create(cls, input_dim, tfid_word, tfid_char, label_map, hand_feature_names=None, hand_mean=None, hand_std=None, loss: LossFunction = None) -> "NumpyModel":
        model = cls(len(label_map))
        if model.nn.layers and isinstance(model.nn.layers[0], DenseLayer):
            model.nn.layers[0].set_input_shape((input_dim,))
            model.nn.layers[0].initialize()

        model.tfidf_word = tfid_word
        model.tfidf_char = tfid_char
        model.label_map = label_map
        model.inverse_label_map = {v: k for k, v in label_map.items()}
        model.hand_feature_names = hand_feature_names
        model.hand_mean = hand_mean
        model.hand_std = hand_std
        model.loss = loss
        return model

    @classmethod
    def load(cls, path: str) -> "NumpyModel":
        try:
            with gzip.open(path, "rb") as file:
                return pickle.load(file)
        except (OSError, EOFError):
            with open(path, "rb") as file:
                return pickle.load(file)

    def save(self, path: str):
        try:
            with gzip.open(path, "wb") as file:
                pickle.dump(self, file)
        except OSError:
            with open(path, "wb") as file:
                pickle.dump(self, file)

    def predict(self, texts, y=None):
        if isinstance(texts, np.ndarray):
            features = texts.astype(np.float32, copy=False)
        elif isinstance(texts, list) and (len(texts) == 0 or isinstance(texts[0], str)):
            if self.tfidf_word is None or self.tfidf_char is None:
                raise ValueError("TF-IDF vectorizers are not attached to this model.")

            word_features = self.tfidf_word.transform(texts)
            char_features = self.tfidf_char.transform(texts)
            
            # Extract hand-crafted features if available
            if self.hand_feature_names is not None:
                x_hand = []
                for text in texts:
                    text_clean = preprocess_text(text)
                    feat_dict = extract_features(text, text_clean)
                    feat_vec = [feat_dict.get(name, 0.0) for name in self.hand_feature_names]
                    x_hand.append(feat_vec)
                x_hand = np.array(x_hand, dtype=np.float32)
                
                # Normalize using training statistics
                if self.hand_mean is not None and self.hand_std is not None:
                    x_hand = (x_hand - self.hand_mean) / self.hand_std
                
                features = np.hstack([word_features, char_features, x_hand]).astype(np.float32)
            else:
                features = np.hstack([word_features, char_features]).astype(np.float32)
        else:
            raise TypeError("X must be a numpy feature matrix or a list of raw texts.")

        predictions_probs = self.nn.forward_propagation(features, training=False)
        predictions = predictions_probs.argmax(axis=1)

        if y is not None:
            if isinstance(y, np.ndarray):
                labels = y.astype(np.int32, copy=False)
            elif isinstance(y, list) and (len(y) == 0 or isinstance(y[0], str)):
                if self.class_names is None:
                    raise ValueError("Class names are not attached to this model.")
                labels = np.array([self.label_map[label] for label in y], dtype=np.int32)
            else:
                raise TypeError("y must be a numpy array or a list of labels.")

            total = len(labels)
            correct = np.sum(predictions == labels)
            print(f"Accuracy: {correct}/{total} ({correct/total:.2%})")

        return [self.inverse_label_map[p] for p in predictions]
