# AP — Aprendizagem Profunda (MIA, Grupo 8)

Universidade do Minho — Unidade Curricular de Aprendizagem Profunda.  
Trabalhos desenvolvidos ao longo dos dois módulos da UC, Grupo 8.

## Equipa

| Aluno | Nome |
|---|---|
| PG11605 | Carlos da Mota Bergueira |
| PG59999 | Diego Jefferson Mendes Silva |
| PG42201 | Filipa Araújo Pereira |
| PG7942  | Rui Manuel Martins Marques Rodrigues |

---

## Módulo 1 — Classificação de Texto (`modulo-1/`)

Classificação multiclasse de notícias (dataset AG News) com quatro abordagens
progressivas: de uma DNN em NumPy puro até fine-tuning de BERT/RoBERTa.

### Estrutura

```
modulo-1/
├── src/
│   ├── models/          # 4 implementações (NumPy DNN, PyTorch DNN, Transformer, BERT)
│   ├── prepare/         # utilitários de montagem e pré-processamento do dataset
│   ├── utils/           # helpers partilhados (treino, métricas, device)
│   ├── data/            # CSVs locais (ag_news, subm1/2/3, exemplos anotados)
│   ├── eval_*.py        # scripts de avaliação nos conjuntos rotulados
│   └── analise_dataset.ipynb
├── models/              # artefactos treinados (.pt, .pkl.gz) + resultados
├── docs/article/        # relatório LaTeX + imagens
└── Relatorio/           # versão Word do relatório
```

Os artefactos de submissão (previsões CSV + notebooks) encontram-se na raiz do
repositório em `Subm1/`, `Subm2/` e `Subm3/`.

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Treino dos modelos

Executar a partir de `modulo-1/src/`:

```bash
cd modulo-1/src
```

```bash
# NumPy DNN
python -c "from models.dnn.train import TrainNumpy; TrainNumpy.train()"

# PyTorch DNN
python -c "from models.pytorch.train import PyTorchTrainer; PyTorchTrainer.train()"

# Transformer custom
python -c "from models.transformer.train import TransformerTrainer; TransformerTrainer.train()"

# BERT/RoBERTa fine-tuning
python -c "from models.bert.train import BertTrainer; BertTrainer.train()"
```

### Avaliação

```bash
python eval_dataset_exemplo_models.py
python eval_subm1_models.py
python eval_subm2_models.py
```

Os resultados são escritos em `modulo-1/models/results/`.

---

## Módulo 2 — Classificação de Imagens ERCP (`modulo-2/`)

Classificação multiclasse de imagens endoscópicas ERCP (dataset MIQR-CC, 4
classes: Biliary Leaks, Lithiasis, Normal, Stricture) com quatro arquiteturas
CNN pré-treinadas. Melhor resultado: **DenseNet121 — F1 macro = 0.7076**.

### Estrutura

```
modulo-2/
├── notebooks/
│   ├── DENSENET.ipynb       # notebook final DenseNet121
│   ├── RESNET.ipynb         # notebook final ResNet50
│   ├── MOBILENET.ipynb      # notebook final MobileNetV2
│   ├── EFICIENTNET.ipynb    # notebook final EfficientNet-B7
│   └── old-versions/        # histórico de versões intermédias (v1–v7 por arq.)
├── docs/
│   ├── relatorio/           # relatório técnico final (LaTeX LLNCS + PDF)
│   └── apresentacao/        # apresentação Beamer 16:9 (LaTeX + PDF)
└── requirements.txt
```

### Notebooks por arquitectura

| Notebook | Arquitectura | Melhor F1 macro |
|---|---|---|
| `DENSENET.ipynb` | DenseNet121 | **0.7076** |
| `RESNET.ipynb` | ResNet50 | 0.6647 |
| `EFICIENTNET.ipynb` | EfficientNet-B7 | 0.5557 |
| `MOBILENET.ipynb` | MobileNetV2 | 0.5558 |

Versões intermédias (histórico experimental completo) disponíveis em
`modulo-2/notebooks/old-versions/`.

### Instalação

O módulo 2 usa [`uv`](https://docs.astral.sh/uv/) para gestão de dependências.

```bash
# Instalar uv (se necessário)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Criar ambiente e instalar dependências (com hash verification)
uv sync

# Alternativa via pip
pip install -r modulo-2/requirements.txt
```

> Requer Python ≥ 3.10. Em macOS usa MPS (Apple Silicon); em Linux usa CUDA 13.

### Documentação

| Ficheiro | Descrição |
|---|---|
| `modulo-2/docs/relatorio/relatorio_AP.tex` | Relatório técnico (LLNCS/Springer) |
| `modulo-2/docs/apresentacao/apresentacao.tex` | Apresentação Beamer (10 min, 11 slides) |