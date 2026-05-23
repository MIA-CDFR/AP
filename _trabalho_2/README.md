# Classificação de Imagens ERCP com CNNs Profundas

Trabalho prático de Aprendizagem Profunda — Módulo 2.  
Comparação de quatro arquitecturas CNN (DenseNet121, ResNet50, MobileNetV2, EfficientNet-B7) para classificação multiclasse de imagens endoscópicas ERCP do dataset MIQR-CC.

**Dataset:** 1 067 treino / 234 validação / 267 teste — 4 classes (Biliary Leaks, Lithiasis, Normal, Stricture)  
**Melhor resultado:** DenseNet121 v100 — F1 macro = **0.7076**, Accuracy = **0.7266**

---

## Instalação de dependências

```bash
# Instalar dependências via pip (sem verificação de hashes)
pip install -r requirements.txt
```

> **Nota:** o ficheiro `requirements.txt` foi gerado automaticamente por `uv export -o requirements.txt` e inclui hashes SHA-256 para reprodutibilidade total. O ambiente requer Python ≥ 3.10. Em macOS utiliza MPS (Apple Silicon); em Linux usa CUDA 13.

---

## Estrutura dos notebooks

### DenseNet121 — `_CB/`

| Notebook | Descrição |
|---|---|
| `_DENSENET_CB_v0.ipynb` | **Baseline** — pipeline de referência: DenseNet121 pré-treinado, Focal Loss, Adam, 512×512, batch=4 |
| `_DENSENET_CB_v1.ipynb` | v0 com resolução reduzida (224×224) e batch maior (16) para treino mais rápido |
| `_DENSENET_CB_v2.ipynb` | v0 + CLAHE aplicado inline no pipeline de treino (512×512) |
| `_DENSENET_CB_v3.ipynb` | v2 com resolução 224×224 e batch=16 |
| `_DENSENET_CB_v99.ipynb` | **F1Boost** — AdamW + OneCycleLR, Focal Loss + CrossEntropy com class weights e label smoothing, Mixup augmentation, TTA, threshold tuning por classe, early stopping patience=15, gradient clipping |
| `_DENSENET_CB_v100.ipynb` | ⭐ **Melhor modelo** — v99 + CLAHE aplicado offline a todo o dataset antes do treino |
| `_DENSENET_CB_v101.ipynb` | v100 + augmentation extra da classe Biliary_Leaks + early stopping patience=20 |
| `_DENSENET_CB_v102.ipynb` | v101 com resolução original 512×512 |
| `_CB_ENSEMBLE_v0.ipynb` | **Ensemble v0** — combina DenseNet121 + EfficientNet-B7 por soft voting (média de probabilidades) |
| `_CB_ENSEMBLE_v1.ipynb` | **Ensemble v1** — DenseNet v99 + EfficientNet com soft voting |

### ResNet50 — `_DS/`

| Notebook | Descrição |
|---|---|
| `RESNET_final_v1.ipynb` | **Domínio-guiado** — pré-processamento especializado: CropForeground, CenterSpatialCrop(480×480), ScaleIntensityRangePercentiles, máscara circular endoscópica, Dropout(0.3) no classificador |
| `RESNET_final_v2.ipynb` | **Baseline regularizado** — pipeline genérico idêntico à referência com adição de Dropout(0.3), sem especialização de domínio |
| `RESNET_final_v3.ipynb` | **CLAHE + domínio** — v1 com imagens pré-processadas offline com CLAHE; testa o impacto do realce de contraste sobre o pipeline anatómico |

### MobileNetV2 — `_FP/`

| Notebook | Descrição |
|---|---|
| `MOBILENET_v4_CLAHE.ipynb` | CLAHE aplicado offline a todo o dataset (mais eficiente que inline); class weights suavizados, fine-tuning faseado (backbone congelado → desbloqueado), Mixup |
| `MOBILENET_v6.ipynb` | v4 + carregamento RGB com PIL + normalização com ImageNet stats (mean/std) + transforms torchvision — corrige incompatibilidade das versões anteriores com o pré-treino do MobileNetV2 |
| `MOBILENET_v7.ipynb` | v6 + **differential learning rates** (sem fase de backbone congelado): LR=1e-5 nas camadas iniciais, 5e-5 nas finais, 1e-3 no classificador; warm-up nas primeiras 5 épocas |

### EfficientNet-B7 — `_RR/`

| Notebook | Descrição |
|---|---|
| `EFICIENTNET_RR10.ipynb` | Pipeline base EfficientNet-B7 com CLAHE integrado via `Lambda` no MONAI, resolução 256×256 |
| `EFICIENTNET_RR12.ipynb` | v10 + TTA (n=8 augmentações na inferência) + Grad-CAM para interpretabilidade das predições |
| `EFICIENTNET_rr_101.ipynb` | Variante experimental com pipeline de pré-processamento modificado |

---

## Relatório e apresentação

| Ficheiro | Descrição |
|---|---|
| `Relatório/relatorio_AP.tex` | Relatório técnico final (formato LLNCS/Springer) |
| `Relatório/apresentacao.tex` | Apresentação Beamer (16:9, 10 min, 11 slides) |
| `Relatório/guiao.md` | Guião do orador com timing e respostas a perguntas previsíveis |
| `_CB/REPORT_CB_DENSENET.md` | Relatório técnico detalhado da progressão experimental DenseNet |
| `_DS/RELATORIO_TECNICO_RESNET_FINAIS.md` | Relatório técnico detalhado das variantes ResNet |
