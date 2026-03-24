import torch

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BATCH_SIZE = 32


class BertModel:

    def __init__(self, n_classes, checkpoint="roberta-base", label_map=None):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=n_classes
        ).to(self.device)
        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def load(cls, model_path):
        checkpoint_data = torch.load(model_path / "best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        checkpoint_name = checkpoint_data["checkpoint"]
        label_map = checkpoint_data["label_map"]
        model = cls(
            n_classes=len(label_map),
            checkpoint=checkpoint_name,
            label_map=label_map,
        )
        model.model.load_state_dict(checkpoint_data["model_state"])
        model.model.eval()
        return model

    def predict(self, texts, labels=None):
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        preds = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

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
