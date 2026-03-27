import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer, TrainingArguments

from prepare.dataset import get_datasets
from models.transformer.model import TransformModel
from utils.pytorch import torch_utils
from utils.train import compute_metrics_logits
from utils.model import main_folder


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

seq_len: int = 512
model_name: str = "distilbert-base-uncased"


class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, seq_len):
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = TextClsDataset(X_train, y_train, tokenizer, seq_len)
        eval_dataset = TextClsDataset(X_eval, y_eval, tokenizer, seq_len)

        model = TransformModel.create(
            label_map=label_map,
            tokenizer_name=model_name,
            vocab_size=tokenizer.vocab_size,
            pad_idx=tokenizer.pad_token_id or 0,
            seq_len=seq_len,
            d_model=768,        # Increased from 512 for better representation
            num_heads=12,       # Increased from 16, divisible into 768
            num_layers=6,       # Increased from 4 for deeper model
            dropout=0.35,       # Increased from 0.2 for better regularization
        )

        training_args = TrainingArguments(
            output_dir=main_folder / "results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=500,                      # Warmup for stable training
            lr_scheduler_type="cosine",            # Cosine annealing schedule
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
            gradient_accumulation_steps=2,         # Simulate larger batch size
        )

        # Compute class weights for balanced training
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(torch_utils.device)
        # Store weights in model for trainer to use
        model.model.class_weights = class_weights_tensor

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
                "model_state": model.model.state_dict(),
                "tokenizer_name": model_name,
                "pad_token_id": tokenizer.pad_token_id,
                "seq_len": seq_len,
                "label_map": label_map,
                "vocab_size": tokenizer.vocab_size,
            },
            main_folder / "transformer.pt",
        )
        print("Model saved to", main_folder / "transformer.pt")

if __name__ == "__main__":
    TransformerTrainer.train()
