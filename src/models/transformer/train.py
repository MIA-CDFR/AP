import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from prepare.dataset import get_datasets
from models.transformer.model import TransformModel
from utils.dataset import PAD_TOKEN, UNK_TOKEN, encode_text, build_vocab
from utils.pytorch import torch_utils
from utils.train import compute_metrics
from utils.model import main_folder


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

seq_len: int = 64
min_freq: int = 2


class TextClsDataset(Dataset):
    def __init__(self, texts, labels, stoi, seq_len):
        self.input_ids = [encode_text(t, stoi, seq_len) for t in texts]
        self.labels = labels
        self.pad_idx = stoi[PAD_TOKEN]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = (input_ids != self.pad_idx).long()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


class TransformerTrainer:
    @staticmethod
    def train(epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-4, weight_decay: float = 0.01):
        df = get_datasets()

        X_texts = df["Text"].tolist()
        y_labels = df["Label"].tolist()

        unique_labels = sorted(list(set(y_labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        y_labels = [label_map[label] for label in y_labels]

        X_train, X_eval, y_train, y_eval = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

        stoi = build_vocab(X_train, min_freq=min_freq)

        train_dataset = TextClsDataset(X_train, y_train, stoi, seq_len)
        eval_dataset = TextClsDataset(X_eval, y_eval, stoi, seq_len)

        model = TransformModel.create(
            label_map=label_map,
            vocab_size=len(stoi),
            pad_idx=stoi[PAD_TOKEN],
            seq_len=seq_len,
            d_model=128,
            num_heads=4,
            num_layers=2,
            dropout=0.2,
        )

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_strategy="epoch",
            logging_steps=20,
            dataloader_num_workers=4 if torch_utils.device.type == "cpu" else 0,
            dataloader_pin_memory=torch_utils.device.type == "cuda",
            report_to="none",
            fp16=False,
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
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

        torch.save(
            {
                "model_state": model.model.state_dict(),
                "stoi": stoi,
                "pad_token": PAD_TOKEN,
                "unk_token": UNK_TOKEN,
                "seq_len": seq_len,
                "label_map": label_map,
            },
            main_folder / "transformer.pt",
        )
        print("Model saved to", main_folder / "transformer.pt")

if __name__ == "__main__":
    TransformerTrainer.train()
