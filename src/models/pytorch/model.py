import torch
from scipy.sparse import hstack

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
        self.vector = None
        self.vector_word = None
        self.vector_char = None

    @classmethod
    def create(
        cls,
        input_dim,
        label_map: dict[int, str],
        vector_word,
        vector_char=None,
    ) -> "PyTorchModel":
        dnn_model = cls()
        dnn_model.model = DNNClassifier(
            input_dim,
            n_classes=len(label_map)
        ).to(torch_utils.device)
        dnn_model.label_map = label_map
        dnn_model.inverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
        dnn_model.vector = vector_word
        dnn_model.vector_word = vector_word
        dnn_model.vector_char = vector_char
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
        dnn_model.vector_word = checkpoint.get("vector_word", dnn_model.vector)
        dnn_model.vector_char = checkpoint.get("vector_char", None)
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

        if self.vector_word is not None:
            encoded_word = self.vector_word.transform(texts)
            if self.vector_char is not None:
                encoded_char = self.vector_char.transform(texts)
                encoded = hstack([encoded_word, encoded_char], format="csr").toarray()
            else:
                encoded = encoded_word.toarray()
        else:
            encoded = self.vector.transform(texts).toarray()

        input_ids = torch.tensor(encoded, dtype=torch.float).to(torch_utils.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                if self.label_map is None or "Human" not in self.label_map:
                    raise ValueError("Expected 'Human' in label_map for hierarchical DNN prediction.")

                binary_logits, family_logits = outputs
                human_class_index = self.label_map["Human"]
                ai_class_indices = sorted(
                    class_index
                    for label, class_index in self.label_map.items()
                    if label != "Human"
                )
                family_to_class_tensor = torch.tensor(
                    ai_class_indices,
                    dtype=torch.long,
                    device=input_ids.device,
                )

                binary_preds = torch.sigmoid(binary_logits).squeeze(1) >= 0.5
                family_preds = family_logits.argmax(dim=1)
                preds_tensor = torch.full(
                    (input_ids.shape[0],),
                    human_class_index,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                if binary_preds.any():
                    preds_tensor[binary_preds] = family_to_class_tensor[family_preds[binary_preds]]
                preds = preds_tensor.cpu().tolist()
            else:
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
