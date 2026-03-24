import torch

from typing import Any

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.pytorch import torch_utils


BATCH_SIZE = 32


class BertModel:

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_map = None
        self.inverse_label_map = None

    @classmethod
    def from_pretrained(cls, model_name: str, label_map: dict[str, Any]) -> "BertModel":
        bert_model = cls()
        bert_model.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_map)
        ).to(torch_utils.device)
        bert_model.label_map = label_map
        bert_model.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
        return bert_model

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "BertModel":
        bert_model = cls()
        bert_model.tokenizer = AutoTokenizer.from_pretrained(checkpoint["checkpoint"])
        bert_model.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint["checkpoint"],
            num_labels=len(checkpoint["label_map"])
        ).to(torch_utils.device)
        bert_model.label_map = checkpoint["label_map"]
        bert_model.inverse_label_map = {v: k for k, v in bert_model.label_map.items()} if bert_model.label_map else None

        model_state = checkpoint["model_state"]
        # Auto-upcast FP16 weights to FP32 for inference (lossless and safer)
        upcasted_state = {}
        for k, v in model_state.items():
            if torch.is_floating_point(v) and v.dtype == torch.float16:
                upcasted_state[k] = v.float()
            else:
                upcasted_state[k] = v

        bert_model.model.load_state_dict(upcasted_state)
        bert_model.model.eval()
        return bert_model

    @staticmethod
    def load(model_path) -> "BertModel":
        return BertModel.from_checkpoint(
            checkpoint=torch.load(
                model_path,
                map_location=torch_utils.device
            ),
        )

    def predict(self, texts, labels=None):
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(torch_utils.device)

        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        preds = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(torch_utils.device)
                attention_mask = attention_mask.to(torch_utils.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                preds.extend(logits.argmax(dim=1).cpu().numpy())

        pred_labels = [self.inverse_label_map[p] for p in preds] if self.inverse_label_map else preds

        if labels:
            correct = sum(p == l for p, l in zip(pred_labels, labels))
            total = len(labels)
            print(f"Accuracy: {correct}/{total} ({correct/total:.2%})")

        return pred_labels
