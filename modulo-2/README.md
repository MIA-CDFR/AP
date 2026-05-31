# AP — Aprendizagem Profunda (MIA, Grupo 8) — Módulo 2

Universidade do Minho — Unidade Curricular de Aprendizagem Profunda.  
Trabalhos desenvolvidos no **Módulo 2** da UC, Grupo 8 — Classificação de Imagens com Redes Convolucionais.

## Equipa

| Aluno | Nome |
|---|---|
| PG11605 | Carlos da Mota Bergueira |
| PG59999 | Diego Jefferson Mendes Silva |
| PG42201 | Filipa Araújo Pereira |
| PG7942  | Rui Manuel Martins Marques Rodrigues |

---

## Estrutura

```
modulo-2/
├── notebooks/
│   ├── DENSENET.ipynb      # Classificador com DenseNet
│   ├── EFICIENTNET.ipynb   # Classificador com EfficientNet
│   ├── MOBILENET.ipynb     # Classificador com MobileNet
│   ├── RESNET.ipynb        # Classificador com ResNet
│   └── old-versions/       # Versões anteriores dos notebooks
├── docs/                   # Relatórios e apresentações
├── requirements.txt        # Dependências do projeto
└── README.md
```

---

## Configuração do Ambiente

### Pré-requisitos

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (recomendado) **ou** `pip`
- GPU com CUDA (recomendado para treino) — alternativamente Apple Silicon (MPS) ou CPU

### Opção A — com `uv` (recomendado)

```bash
# 1. Instalar uv (se necessário)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Na raiz do workspace (pasta AP/), criar o ambiente virtual e instalar dependências
cd /caminho/para/AP
uv sync

# 3. Activar o ambiente virtual
source .venv/bin/activate
```

### Opção B — com `venv` + `pip`

```bash
# 1. Criar o ambiente virtual
python3.12 -m venv .venv

# 2. Activar o ambiente virtual
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r modulo-2/requirements.txt
```

---

## Executar os Notebooks

### No VS Code

1. Abrir o notebook desejado em `modulo-2/notebooks/`.
2. Clicar em **"Select Kernel"** (canto superior direito).
3. Escolher **"Python Environments…"** → seleccionar `.venv` (o ambiente criado acima).
4. Executar as células com `Shift+Enter` ou usar **"Run All"**.

### Na linha de comandos (Jupyter)

```bash
# Com o ambiente virtual activado:
jupyter notebook modulo-2/notebooks/DENSENET.ipynb
# ou para abrir todos:
jupyter lab modulo-2/notebooks/
```

---

## Notas

- **GPU (CUDA):** o PyTorch usa automaticamente a GPU se disponível. Para verificar:
  ```python
  import torch
  print(torch.cuda.is_available())   # CUDA
  print(torch.backends.mps.is_available())  # Apple Silicon
  ```
- **Apple Silicon (MPS):** suportado; note que `float64` não é suportado em MPS — use `float32`.
- Os pesos pré-treinados são descarregados automaticamente pelo `torchvision` na primeira execução (requer ligação à internet).
