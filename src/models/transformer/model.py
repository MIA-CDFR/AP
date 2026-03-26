import torch

from typing import Any
from transformers import AutoTokenizer

from models.transformer.classifier.transformer import TransformerClassifier

from utils.pytorch import torch_utils



class TransformModel:
    def __init__(self):
        self.model = None
        self.label_map = None
        self.inverse_label_map = None
        self.checkpoint = None

    @classmethod
    def create(
            cls,
            label_map: dict[str, Any],
            tokenizer_name: str,
            vocab_size: int,
            pad_idx: int = 0,
            seq_len: int = 128,
            d_model: int = 512,
            num_heads: int = 16,
            num_layers: int = 4,
            dropout: float = 0.2,
        ) -> "TransformModel":
        transform_model = cls()
        transform_model.model = TransformerClassifier(
            vocab_size=vocab_size,
            n_classes=len(label_map),
            pad_idx=pad_idx,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(torch_utils.device)
        transform_model.label_map = label_map
        transform_model.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
        transform_model.checkpoint = {
            "tokenizer_name": tokenizer_name,
            "seq_len": seq_len,
            "pad_token_id": pad_idx,
            "vocab_size": vocab_size,
        }
        return transform_model

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "TransformModel":
        transform_model = cls()
        transform_model.model = TransformerClassifier(
            vocab_size=checkpoint["vocab_size"],
            n_classes=len(checkpoint["label_map"]),
            pad_idx=checkpoint.get("pad_token_id") or 0,
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
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint["tokenizer_name"])
        seq_len = self.checkpoint["seq_len"]

        self.model.eval()

        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(torch_utils.device)
        attention_mask = encoded["attention_mask"].to(torch_utils.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = logits.argmax(dim=1).cpu().tolist()

        pred_labels = [self.inverse_label_map[p] for p in preds] if self.inverse_label_map else preds

        if labels is not None:
            labels_cmp = labels
            if len(labels_cmp) > 0 and not isinstance(labels_cmp[0], str) and self.inverse_label_map is not None:
                labels_cmp = [self.inverse_label_map[int(x)] for x in labels_cmp]
            correct = sum(p == l for p, l in zip(pred_labels, labels_cmp))
            total = len(labels_cmp)
            print(f"Accuracy: {correct}/{total} ({correct/total:.2%})")

        return pred_labels
