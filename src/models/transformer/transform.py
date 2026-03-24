import numpy as np
import torch

from typing import Any

from models.pytorch.prepare.feature import preprocess_text, preprocess_text_clean, build_handcrafted_matrix
from models.transformer.classifier.transformer import TransformerClassifier

from utils.pytorch import torch_utils


class TransformModel:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_map = None
        self.inverse_label_map = None
        self.checkpoint = None

    @classmethod
    def create(cls, vectorizer, input_dim, label_map, seq_len) -> "TransformModel":
        transform_model = cls()
        transform_model.vectorizer = vectorizer
        transform_model.model = TransformerClassifier(
            input_dim,
            len(label_map),
            seq_len=seq_len,
        ).to(torch_utils.device)
        transform_model.label_map = label_map
        transform_model.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
        return transform_model

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "TransformModel":
        transform_model = cls()
        transform_model.vectorizer = checkpoint["vectorizer"]
        transform_model.model = TransformerClassifier(
            checkpoint["input_dim"],
            len(checkpoint["label_map"]),
            seq_len=checkpoint["seq_len"],
        ).to(torch_utils.device)
        transform_model.label_map = checkpoint["label_map"]
        transform_model.inverse_label_map = {v: k for k, v in transform_model.label_map.items()} if transform_model.label_map is not None else None
        transform_model.checkpoint = checkpoint
        transform_model.model.load_state_dict(checkpoint["model_state"])
        transform_model.model.eval()
        return transform_model

    @staticmethod
    def load(model_path) -> "TransformModel":
        return TransformModel.from_checkpoint(
            checkpoint=torch.load(
                model_path,
                map_location=torch_utils.device,
                weights_only=False,
            ),
        )

    def predict(self, texts, labels=None):
        texts_clean = [preprocess_text(t) for t in texts]
        clean_light = [preprocess_text_clean(t) for t in texts]


        X_tfidf = self.vectorizer.transform(texts_clean)

        X_hand, _ = build_handcrafted_matrix(texts, clean_light)

        mean = self.checkpoint["hand_mean"]
        std = self.checkpoint["hand_std"]

        X_hand = (X_hand - mean) / std

        X = np.hstack([X_tfidf.toarray(), X_hand])

        # PSEUDO-SEQUÊNCIA (igual ao treino)
        seq_len = self.checkpoint["seq_len"]

        pad_size = (seq_len - (X.shape[1] % seq_len)) % seq_len
        if pad_size > 0:
            X = np.hstack([X, np.zeros((X.shape[0], pad_size))])

        embed_dim = X.shape[1] // seq_len
        X = X.reshape(-1, seq_len, embed_dim)

        # Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(torch_utils.device)

        # Prever
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()

        # Converter labels
        pred_labels = [self.inverse_label_map[p] for p in preds]

        if labels:
            correct = sum(p == l for p, l in zip(pred_labels, labels))
            total = len(labels)
            print(f"Accuracy: {correct}/{total} ({correct/total:.2%})")

        return pred_labels
