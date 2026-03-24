from pathlib import Path
import numpy as np
import torch

from models.pytorch.prepare.feature import preprocess_text, preprocess_text_clean, build_handcrafted_matrix
from models.transform.transformers import TransformerClassifier


class TransformModel:
    def __init__(self, input_dim, n_classes, seq_len_model, vectorizer=None, label_map=None, checkpoint=None):
        self.vectorizer = vectorizer
        self.model = TransformerClassifier(input_dim, n_classes, seq_len=seq_len_model).to(self.device)
        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()} if label_map is not None else None
        self.checkpoint = checkpoint

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def load(cls, model_path):
        checkpoint_data = torch.load(model_path / "transformer.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"), weights_only=False)
        input_dim = checkpoint_data["input_dim"]
        n_classes = len(checkpoint_data["label_map"])
        seq_len_model = checkpoint_data["seq_len"]
        model = cls(
            input_dim=input_dim,
            n_classes=n_classes,
            seq_len_model=seq_len_model,
            vectorizer=checkpoint_data["vectorizer"],
            label_map=checkpoint_data["label_map"],
            checkpoint=checkpoint_data
        )
        model.model.load_state_dict(checkpoint_data["model_state"])
        model.model.eval()
        return model

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
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

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
