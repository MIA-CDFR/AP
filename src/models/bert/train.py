import torch

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

from prepare.dataset import get_datasets
from models.bert.model import BertModel
from utils.pytorch import torch_utils
from utils.train import compute_metrics_logits
from utils.model import main_folder


MODEL_NAME = "roberta-base"

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BertTrainer:

    @staticmethod
    def train(epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5, weight_decay: float = 0.01, max_length: int = 256):
        df = get_datasets()

        X_texts = df["Text"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        X_train, X_eval, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

        model = BertModel.from_pretrained(
            MODEL_NAME,
            label_map=label_map
        )

        tokenized_train = model.tokenizer(X_train, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        tokenized_eval = model.tokenizer(X_eval, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

        train_dataset = TextClassificationDataset(tokenized_train, y_train)
        eval_dataset = TextClassificationDataset(tokenized_eval, y_eval)

        training_args = TrainingArguments(
            output_dir=main_folder / "results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_strategy="epoch",
            logging_steps=10,
            dataloader_num_workers=4 if torch_utils.device.type == "cpu" else 0,
            dataloader_pin_memory=torch_utils.device.type == "cuda",
            gradient_accumulation_steps=2,
            report_to="none",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_logits,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

        torch.save(
            {
                "model_name": MODEL_NAME,
                "model_state": model.model.state_dict(),
                "tokenizer": model.tokenizer,
                "label_map": label_map,
            },
            main_folder / "bert.pt",
        )

        print("Model saved to", main_folder / "bert.pt")

if __name__ == "__main__":
    BertTrainer.train()
