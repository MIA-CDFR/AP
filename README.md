# AP - Text Classification (MIA/AP)

Projeto de Aprendizagem Profunda (Universidade do Minho, Grupo 8) para classificacao de texto com multiplas abordagens:

- NumPy DNN
- PyTorch DNN
- Transformer custom
- BERT/RoBERTa fine-tuning

## Team

| Aluno | Nome |
|----------|------------|
| PG11605  | Carlos da Mota Bergueira |
| PG59999  | Diego Jefferson Mendes Silva |
| PG42201  | Filipa Araujo Pereira |
| PG7942   | Rui Manuel Martins Marques Rodrigues |

## Setup

Prerequisites:

- Python 3.12+
- pip

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Overview

### Root files

- `pyproject.toml`: project metadata and dependencies.
- `requirements.txt`: pinned dependency export.
- `README.md`: this guide.

### Main folders (excluding folders starting with `_`)

- `src/`: core codebase for dataset preparation, model training, and evaluation scripts.
- `src/models/`: the 4 model implementations and notebooks used during experiments.
- `src/prepare/`: dataset assembly utilities (`get_datasets`) and preprocessing pipeline.
- `src/utils/`: shared helpers for training, dataset processing, metrics, and device/model utilities.
- `src/data/`: local CSV datasets (`ag_news`, `subm1`, `subm2`, `subm3`, revealed labels, examples).
- `models/`: trained model artifacts (`bert.pt`, `pytorch-dnn.pt`, `transformer.pt`, and NumPy model files).
	- `models/results/`: evaluation outputs and transformer training checkpoints.
- `docs/article/`: report assets (LaTeX source and images).
- `Subm1/`, `Subm2/`, `Subm3/`: project submission deliverables (CSV predictions + notebooks) for each of the three rounds.

## Training The Four Models

Important: run training commands from `src/`, because model save paths are defined relative to that working directory.

```bash
cd src
```

### 1) NumPy DNN (`models/numpy-dnn.pkl.gz`)

```bash
python -c "from models.dnn.train import TrainNumpy; h = TrainNumpy.train(); print('epochs:', len(h['loss']))"
```

What it does briefly:

- builds word TF-IDF (+ optional char/features)
- trains a feedforward DNN implemented in NumPy
- applies early stopping
- saves to `../models/numpy-dnn.pkl.gz`

### 2) PyTorch DNN (`models/pytorch-dnn.pt`)

```bash
python -c "from models.pytorch.train import PyTorchTrainer; h = PyTorchTrainer.train(); print('epochs:', len(h['train_loss']))"
```

What it does briefly:

- builds word + char TF-IDF features
- trains a PyTorch classifier with Adam and early stopping
- saves to `../models/pytorch-dnn.pt`

### 3) Transformer custom (`models/transformer.pt`)

```bash
python -c "from models.transformer.train import TransformerTrainer; h = TransformerTrainer.train(); print('epochs:', len(h['epoch']))"
```

What it does briefly:

- tokenizes text with Hugging Face tokenizer
- trains the custom transformer classifier with `Trainer`
- saves checkpoint info/state dict to `../models/transformer.pt`

### 4) BERT/RoBERTa fine-tuning (`models/bert.pt`)

```bash
python -c "from models.bert.train import BertTrainer; h = BertTrainer.train(); print('epochs:', len(h['epoch']))"
```

What it does briefly:

- loads `roberta-base`
- fine-tunes for multiclass classification via Hugging Face `Trainer`
- saves to `../models/bert.pt`

## Quick Intro: How To Use Trained Models

### Evaluate all 4 models on known labeled sets

Run from `src/`:

```bash
python eval_dataset_exemplo_models.py
python eval_subm1_models.py
python eval_subm2_models.py
```

Outputs are written under `../models/results/`.

### Predict from Python

Example for BERT:

```python
from models.bert.model import BertModel

model = BertModel.load("../models/bert.pt")
preds = model.predict(["Example text to classify", "Another sentence"])
print(preds)
```

The same usage pattern exists for:

- `models.dnn.model.NumpyModel`
- `models.pytorch.model.PyTorchModel`
- `models.transformer.model.TransformModel`

## Notes About Submissions

- `Subm1/`: artifacts for submission round 1.
- `Subm2/`: artifacts for submission round 2.
- `Subm3/`: artifacts for submission round 3.

Each folder contains at least one notebook (`.ipynb`) and exported prediction file(s) (`.csv`) used in delivery.