import torch

from models.pytorch.classifiers import DNNClassifier

from utils.dataset import encode_text
from utils.pytorch import torch_utils

class PyTorchModel:
    def __init__(self):
        self.model = None
        self.label_map = None
        self.inverse_label_map = None
        self.stoi = None
        self.seq_len = None

    @classmethod
    def create(cls, input_dim, label_map: dict[int, str], vector) -> "PyTorchModel":
        dnn_model = cls()
        dnn_model.model = DNNClassifier(
            input_dim,
            n_classes=len(label_map)
        ).to(torch_utils.device)
        dnn_model.label_map = label_map
        dnn_model.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
        dnn_model.vector = vector
        return dnn_model

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "PyTorchModel":
        dnn_model = cls()
        dnn_model.model = DNNClassifier(
            checkpoint["input_dim"],
            n_classes=len(checkpoint["label_map"])
        ).to(torch_utils.device)
        dnn_model.label_map = checkpoint["label_map"]
        dnn_model.inverse_label_map = {v: k for k, v in checkpoint["label_map"].items()} if checkpoint["label_map"] else None
        dnn_model.vector = checkpoint.get("vector", None)
        dnn_model.model.load_state_dict(checkpoint["model_state"])
        return dnn_model
    
    @staticmethod
    def load(path) -> "PyTorchModel":
        checkpoint = torch.load(
            path,
            map_location=torch_utils.device,
            weights_only=False,
        )
        return PyTorchModel.from_checkpoint(checkpoint)

    def predict(self, texts, labels=None):
        self.model.eval()

        encoded = self.vector.transform(texts).toarray()
        input_ids = torch.tensor(encoded, dtype=torch.float).to(torch_utils.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
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
