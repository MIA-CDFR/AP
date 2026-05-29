import torch

from typing import Any

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

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
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(label_map),
            problem_type="single_label_classification",
            label2id=label_map,
            id2label={v: k for k, v in label_map.items()},
        )
        bert_model.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        ).to(torch_utils.device)
        bert_model.label_map = label_map
        bert_model.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
        return bert_model

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "BertModel":
        bert_model = cls()
        bert_model.tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
        bert_model.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint["model_name"],
            config=AutoConfig.from_pretrained(
                checkpoint["model_name"],
                num_labels=len(checkpoint["label_map"]),
                problem_type="single_label_classification",
                label2id=checkpoint["label_map"],
                id2label={v: k for k, v in checkpoint["label_map"].items()},
            ),
            ignore_mismatched_sizes=True,
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
                map_location=torch_utils.device,
                weights_only=False,
            ),
        )

    def predict(self, texts, labels=None, max_length: int = 256, batch_size: int = BATCH_SIZE):
        self.model.eval()

        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
                preds.extend(logits.argmax(dim=1).cpu().tolist())

        pred_labels = [self.inverse_label_map[p] for p in preds] if self.inverse_label_map else preds

        if labels is not None:
            gold = labels
            if self.label_map and len(gold) > 0 and not isinstance(gold[0], str):
                gold = [self.inverse_label_map[int(x)] for x in gold]
            correct = sum(p == l for p, l in zip(pred_labels, gold))
            total = len(gold)
            print(f"Accuracy: {correct}/{total} ({correct/total:.2%})")

        return pred_labels
