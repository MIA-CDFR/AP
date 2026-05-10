# DenseNet — Classificação ERCP: Experiências e Resultados

Trabalho desenvolvido sobre o notebook base fornecido pelo docente (`DENSENET.ipynb`), com o objetivo de melhorar o **F1-score macro** na classificação multiclasse do dataset ERCP.

```
Narrativa para 2 min de apresentação:
https://claude.ai/share/3462db15-e34b-4861-965e-1e09f660fcac

Análise técnica aprofundada:
https://claude.ai/share/c6ea5523-84fd-4dff-adb5-b7609274051e

Apresentação/narrativa para 2 minutos:
https://docs.google.com/presentation/d/156GvGMaEITIIt7Ch0eDGXBNAAiRb5Qws2y1Q2uOVNK4/edit?usp=sharing
```

---

## Dataset

| Split | Amostras | Biliary Leaks | Lithiasis | Normal | Stricture |
|-------|----------|---------------|-----------|--------|-----------|
| Train | 1 067    | 110           | 505       | 197    | 255       |
| Val   | 234      | 24            | 98        | 59     | 53        |
| Test  | 267      | 17            | 123       | 43     | 84        |

O dataset é **desequilibrado**: Lithiasis representa ~47% do treino, Biliary Leaks apenas ~10%.

---

## Arquitetura base

- **Modelo:** DenseNet121 pré-treinado (ImageNet), adaptado para 4 classes
- **Otimizador base:** Adam, lr=1e-4
- **Loss base:** CrossEntropyLoss
- **Scheduler:** CosineAnnealingLR
- **Early stopping:** patience=10, restauro do melhor checkpoint
- **Augmentações base:** RandRotate (±15°), RandZoom (0.9–1.1), RandAdjustContrast, RandGaussianNoise

---

## Versões e variações testadas

### v0 — Baseline fiel

Reprodução controlada do notebook original, com hiperparâmetros centralizados e versionamento automático dos modelos.

| Parâmetro       | Valor   |
|-----------------|---------|
| Resolução       | 512×512 |
| Batch size      | 4       |
| Épocas (melhor) | 9       |

### v1 — Resolução 224px + batch 16

Hipótese: 224×224 é a resolução nativa do ImageNet (pré-treino do DenseNet121); batches maiores estabilizam os gradientes.

| Parâmetro       | Valor   |
|-----------------|---------|
| Resolução       | 224×224 |
| Batch size      | 16      |
| Épocas (melhor) | 7       |

Efeito colateral positivo: tempo de treino ~5× mais rápido (410s vs 2091s).

### v2 — CLAHE em 512px

Adição de CLAHE (*Contrast Limited Adaptive Histogram Equalization*, `clipLimit=2.0`, `tileGridSize=(8,8)`, `prob=0.7`) como augmentação de treino. As imagens ERCP têm contraste variável; o CLAHE normaliza localmente, ajudando o modelo a focar nas estruturas relevantes.

| Parâmetro       | Valor   |
|-----------------|---------|
| Resolução       | 512×512 |
| Batch size      | 4       |
| Épocas (melhor) | 14      |

### v3 — CLAHE + 224px + batch 16

Combinação de v1 e v2, para avaliar se os dois eixos se complementam.

| Parâmetro       | Valor   |
|-----------------|---------|
| Resolução       | 224×224 |
| Batch size      | 16      |
| Épocas (melhor) | 7       |

### v99 — F1Boost

Versão sem restrições, atacando o problema identificado nas versões anteriores: o desequilíbrio de classes. Melhorias introduzidas:

- **AdamW + OneCycleLR** — convergência mais rápida e regularização implícita
- **FocalLoss + CrossEntropy com class weights** — penalização focada nas classes sub-representadas
- **Class weights calculados dinamicamente** — `total / (num_classes × count_per_class)`; peso de Biliary Leaks = 2.43
- **Mixup augmentation** — regularização forte, reduz overfitting
- **Augmentações mais agressivas** — flips H/V, rotação até ±30°, RandShiftIntensity, RandScaleIntensity
- **Test-Time Augmentation (TTA)** — média de 4 variantes na inferência (original, flip H, flip V, rotação)
- **Threshold tuning por classe** — escala de logits optimizada no val set antes de avaliar o test
- **Early stopping patience=15** com restauro dos melhores pesos
- **Gradient clipping** para estabilidade do treino

| Parâmetro       | Valor   |
|-----------------|---------|
| Resolução       | 512×512 |
| Batch size      | 8       |
| Épocas (melhor) | 19      |
| Tempo de treino | ~6.4h   |

---

## Resultados no test set

### F1-score macro

| Versão | F1 macro | Accuracy | Δ vs v0  | Notas                        |
|--------|----------|----------|----------|------------------------------|
| v0     | 0.545    | 0.618    | —        | Baseline                     |
| v1     | 0.513    | 0.693    | −0.032   | Biliary Leaks: F1=0.000      |
| v2     | 0.549    | 0.727    | +0.004   | Biliary Leaks: F1=0.000      |
| v3     | 0.525    | 0.693    | −0.020   | Biliary Leaks: F1=0.000      |
| **v99**| **0.647**| 0.674    | **+0.102**| Melhor resultado             |

### F1 por classe — v0 vs v99

| Classe        | Suporte | v0 F1 | v99 F1 | Δ      |
|---------------|---------|-------|--------|--------|
| Biliary Leaks | 17      | 0.438 | 0.579  | +0.141 |
| Lithiasis     | 123     | 0.708 | 0.708  | +0.000 |
| Normal        | 43      | 0.488 | 0.615  | +0.127 |
| Stricture     | 84      | 0.546 | 0.684  | +0.138 |

### Matrizes de confusão

**v0 (baseline)**
```
                  Biliary  Lithi  Normal  Stricture
Biliary_Leaks  [    7       2       7        1   ]
Lithiasis      [    1     102      12        8   ]
Normal         [    3      17      20        3   ]
Stricture      [    4      44       0       36   ]
```

**v99 (melhor)**
```
                  Biliary  Lithi  Normal  Stricture
Biliary_Leaks  [   11       0       5        1   ]
Lithiasis      [    4      85      22       12   ]
Normal         [    1       7      32        3   ]
Stricture      [    5      25       2       52   ]
```

---

## Análise e conclusões

### O que as versões v1–v3 revelaram

A redução de resolução para 224×224 (v1, v3) **não melhorou o F1 macro**, apesar de acelerar o treino 5×. Pior: a classe Biliary Leaks (a mais rara) deixou de ser detectada — F1=0.000 em v1, v2 e v3. O CLAHE (v2) trouxe uma subida marginal de +0.004. Combinar os dois eixos (v3) não somou os efeitos.

**Conclusão intermédia:** o problema não era a resolução nem o contraste. Era o desequilíbrio de classes.

### Por que a v99 funcionou

A v0 ignorava Biliary Leaks porque a classe minoritária (110 amostras de treino) não tinha peso suficiente no gradiente. A v99 atacou este problema em três frentes simultâneas:

1. **Loss ponderada** — class weights e FocalLoss forçam o modelo a prestar atenção às classes raras
2. **Regularização** — Mixup e augmentações agressivas evitam overfitting nas classes maioritárias
3. **Inferência robusta** — TTA e threshold tuning aumentam a sensibilidade às classes sub-representadas

O resultado foi um ganho de **+10 pontos percentuais** de F1 macro, com Biliary Leaks a passar de 0.44 para 0.58.

### Conclusão

Num problema de classificação médica com classes desequilibradas, **a escolha da função de perda e as técnicas de treino valem muito mais do que optimizar o pré-processamento**. Melhorar a resolução ou adicionar CLAHE são ajustes incrementais; lidar com o desequilíbrio é estrutural.

---

## Estrutura dos ficheiros

```
.
├── DENSENET.ipynb                          # Notebook base (docente)
├── _DENSENET_CB_v0.ipynb                   # Baseline fiel
├── _DENSENET_CB_v1.ipynb                   # 224px + batch 16
├── _DENSENET_CB_v2.ipynb                   # CLAHE (512px)
├── _DENSENET_CB_v3.ipynb                   # CLAHE + 224px + batch 16
├── _DENSENET_CB_v99_Claude_Carta_Branca.ipynb  # F1Boost
├── models/
│   └── MIA_AP_my_models/
│       ├── densenet_v0_20260510_100039.pth
│       ├── densenet_v0_20260510_100039.pth_cm.png
│       ├── densenet_v0_20260510_100039.pth.png
│       ├── densenet_v1_20260510_103929.pth
│       ├── densenet_v1_20260510_103929.pth_cm.png
│       ├── densenet_v1_20260510_103929.pth.png
│       ├── densenet_v2_20260510_104725.pth
│       ├── densenet_v2_20260510_104725.pth_cm.png
│       ├── densenet_v2_20260510_104725.pth.png
│       ├── densenet_v3_20260510_112723.pth
│       ├── densenet_v3_20260510_112723.pth_cm.png
│       ├── densenet_v3_20260510_112723.pth.png
│       ├── densenet_v99_20260509_161534.pth
│       ├── densenet_v99_20260509_161534.pth_cm.png
│       └── densenet_v99_20260509_161534.pth.png
└── README_DENSENET.md                      # Este ficheiro
```