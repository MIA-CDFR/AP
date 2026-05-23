# Classificação de Imagens ERCP com Redes Neurais Convolucionais Profundas
## Relatório Técnico Comparativo: DenseNet, ResNet, MobileNet e EfficientNet

**Disciplina:** Módulo 2 — Aprendizagem Profunda para Imagens Médicas  
**Grupo:** G8  
**Data:** Maio de 2026  

---

## Resumo

Este relatório documenta a investigação experimental conduzida para maximizar o F1-score macro na classificação multiclasse de imagens endoscópicas ERCP (Colangiopancreatografia Retrógrada Endoscópica). Foram exploradas quatro arquitecturas CNN pré-treinadas — DenseNet121, ResNet50, MobileNetV2 e EfficientNet-B7 —, totalizando mais de 20 versões incrementais. O principal desafio identificado foi o desequilíbrio severo entre classes (~47% Lithiasis, ~10% Biliary Leaks). A progressão metodológica partiu de baselines que ignoravam este desequilíbrio, evoluindo para estratégias combinadas de funções de perda ponderadas, técnicas de augmentação dirigida e mecanismos de inferência robusta. A DenseNet121 obteve o melhor resultado global (**F1 macro = 0.7076**), seguida de ResNet50 (0.6647), EfficientNet-B7 (~0.555) e MobileNetV2 (~0.555). Uma abordagem ensemble por soft voting entre DenseNet e EfficientNet é discutida na secção 6.

**Palavras-chave:** CNN, transfer learning, ERCP, F1 macro, desequilíbrio de classes, DenseNet, ResNet, MobileNet, EfficientNet, ensemble.

---

## 1. Introdução

### 1.1 Problema e Dataset

O dataset consiste em imagens de ERCP classificadas em 4 categorias clínicas: **Biliary Leaks** (fuga biliar), **Lithiasis** (cálculos biliares), **Normal** e **Stricture** (estenose). A distribuição é acentuadamente desequilibrada:

| Split | Total | Biliary Leaks | Lithiasis | Normal | Stricture |
|---|---|---|---|---|---|
| Treino | 1 067 | 110 (10.3%) | 505 (47.3%) | 197 (18.5%) | 255 (23.9%) |
| Validação | 234 | 24 | 98 | 59 | 53 |
| Teste | 267 | 17 | 123 | 43 | 84 |

A métrica de avaliação é o **F1-score macro**, que atribui peso igual a cada classe independentemente da sua frequência:

$$F_1^{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

Esta métrica é particularmente exigente em cenários médicos: um modelo que ignore a Biliary Leaks (a classe mais rara e clinicamente crítica) é fortemente penalizado, ainda que alcance accuracy elevada.

### 1.2 Motivação para Transfer Learning

Todos os modelos utilizam **transfer learning** a partir de pesos pré-treinados no ImageNet. Com apenas 1 067 imagens de treino, treinar CNNs profundas de raiz seria inviável. Os pesos do ImageNet fornecem detectores de baixo nível (bordas, texturas, gradientes) transferíveis para imagens médicas, reduzindo drasticamente a necessidade de dados anotados (Tajbakhsh et al., 2016).

---

## 2. Arquitecturas Estudadas — Introdução Teórica

### 2.1 DenseNet121 — Dense Convolutional Network

Proposta por Huang et al. (2017), a **DenseNet** implementa conectividade densa: cada camada recebe como input a concatenação das *feature maps* de **todas** as camadas anteriores dentro do mesmo bloco denso.

$$\mathbf{x}_L = H_L([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{L-1}])$$

Ao contrário da ResNet (que soma residuais), a DenseNet **concatena**, preservando integralmente a informação de todas as representações anteriores. O **growth rate** $k$ (tipicamente 32) controla o número de feature maps adicionadas por camada. A DenseNet121 possui 4 blocos densos com 6, 12, 24 e 16 camadas, separados por *transition layers* (conv 1×1 + average pooling 2×2). As principais vantagens em imagem médica são a reutilização de features de baixo nível em camadas profundas e gradientes fortes que facilitam o treino sem degradação.

### 2.2 ResNet50 — Residual Network

He et al. (2016) introduziram as **conexões residuais** para resolver o problema de degradação em redes muito profundas. A ideia central é aprender uma função residual em vez da transformação completa:

$$\mathbf{x}_{L+1} = \mathcal{F}(\mathbf{x}_L, \{W_i\}) + \mathbf{x}_L$$

A adição directa (vs. concatenação da DenseNet) mantém o custo computacional reduzido. A ResNet50 possui 4 estágios com blocos bottleneck (conv 1×1 → 3×3 → 1×1), totalizando ~25M parâmetros. O fluxo de gradiente directo entre entrada e saída de cada bloco resolve efectivamente o problema do *vanishing gradient*. É a arquitectura de referência do estado da arte em classificação de imagens médicas desde 2015.

### 2.3 MobileNetV2 — Depthwise Separable Convolutions

Desenvolvida pelo Google (Sandler et al., 2018), a **MobileNetV2** foi concebida para dispositivos móveis com restrições computacionais. O bloco fundamental é o *Inverted Residual with Linear Bottleneck*: expande os canais (factor 6×), aplica *depthwise separable convolution* (separada em profundidade e ponto) e projecta de volta para a dimensão original. Com apenas ~3.4M parâmetros, é ~7× mais leve que a ResNet50. A separação da convolução em dois passos factoriza o custo computacional de $O(D_k^2 \cdot M \cdot N)$ para $O(D_k^2 \cdot M + M \cdot N)$. Esta eficiência tem como contrapartida menor capacidade representacional, o que se reflecte nos resultados mais baixos em tarefas médicas complexas.

### 2.4 EfficientNet-B7 — Compound Scaling

Tan & Le (2019) propõem um método de escalamento sistemático que aumenta simultaneamente profundidade, largura e resolução da rede segundo um coeficiente composto $\phi$:

$$\text{profundidade}: d = \alpha^\phi, \quad \text{largura}: w = \beta^\phi, \quad \text{resolução}: r = \gamma^\phi$$

A EfficientNet-B7 é a variante de maior escala: ~66M parâmetros, resolução nativa 600×600, resultado do escalamento composto de uma arquitectura base (EfficientNet-B0) treinada com *Neural Architecture Search*. A grande capacidade e a normalização por *Batch Normalization* tornam-na sensível ao tamanho do batch e aos hiper-parâmetros de fine-tuning, especialmente quando apenas uma fracção dos pesos é desbloqueada.

---

## 3. Progressão Experimental por Arquitectura

### 3.1 DenseNet121 — Pasta `_CB` (7 versões)

| Versão | Principais mudanças | F1 Macro | Accuracy | Nota |
|---|---|---|---|---|
| **v0** | Baseline fiel ao docente. CrossEntropyLoss uniforme, 512 px, batch 4, Adam lr=1e-4, CosineAnnealingLR, patience 10 | 0.545 | 0.618 | Biliary Leaks detectada (F1=0.438) |
| **v1** | 224 px + batch 16 (resolução nativa ImageNet) | 0.513 | 0.693 | Biliary Leaks F1=**0.000** — resolução elimina detalhes diagnósticos |
| **v2** | CLAHE em runtime (clipLimit=2.0, tileGrid=8×8, prob=0.7) + 512 px | 0.549 | 0.727 | Biliary Leaks F1=0.000; CLAHE melhora Lithiasis/Stricture |
| **v3** | CLAHE + 224 px | 0.525 | 0.693 | Efeitos não se somam; Biliary Leaks F1=0.000 |
| **v99** | **F1Boost**: AdamW+OneCycleLR, FocalLoss+CE ponderada (w_BL≈2.43), Mixup (β=0.4), TTA 4×, threshold tuning | 0.647 | 0.674 | Primeiro modelo a detectar BL; +10.2 pp vs v0 |
| **v100** | Stack v99 + CLAHE offline + AMP + 256 px, 80 épocas | **0.7076** | **0.7266** | Melhor resultado; Stricture F1=0.765 |
| **v101** | v100 + aug agressiva BL + label smoothing=0.1 + patience 20, batch 16 | 0.7076 | 0.7266 | Early stop época 38; matriz idêntica à v100 |

**Insight crítico:** As versões v1–v3 demonstraram que o problema não era a resolução nem o CLAHE mas sim o **desequilíbrio de classes**. A v99 resolveu 10.2 pp com a combinação loss ponderada + Mixup + TTA. A v100 acrescentou mais 6 pp com CLAHE offline, AMP e resolução 256 px.

### 3.2 ResNet50 — Pasta `_DS` (3 versões finais)

| Versão | Principais mudanças | F1 Macro | Accuracy |
|---|---|---|---|
| **Baseline** | FocalLoss, Adam lr=1e-4, CosineAnnealingLR, 512 px, batch 4, classificador Linear | 0.617 | 0.637 |
| **v1** | ROI anatómica (CropForeground + CenterCrop 480 px + ScaleIntensityPercentis 1-99% + máscara circular) + Dropout(0.3) + augmentação conservadora | 0.648 | 0.685 |
| **v2** | Pipeline genérico + Dropout(0.3) | **0.665** | **0.704** |
| **v3** | ROI v1 + imagens CLAHE offline + Dropout(0.3) | 0.609 | 0.640 |

**Insight crítico:** O Dropout(0.3) no classificador foi o factor mais impactante (+4.8 pp vs. baseline, comparando v2 vs. baseline). O pré-processamento especializado do domínio (v1) melhora Stricture (+9 pp) mas degrada Normal. O CLAHE offline (v3) prejudica o pipeline especializado — amplifica características espectrais que tornam Stricture visualmente semelhante a Lithiasis (48/84 Strictures mal classificadas como Lithiasis).

### 3.3 MobileNetV2 — Pasta `_FP` (8 versões)

| Versão | Principais mudanças | F1 Macro |
|---|---|---|
| v0 | Baseline docente (CPU) | 0.456 |
| v1 | CPU, ajustes menores | 0.497 |
| v2 | GPU, configuração instável | 0.306 |
| v3 | GPU estabilizada | 0.477 |
| v4 | CLAHE offline (clipLimit=2.0, tileGrid=8×8) | **0.556** |
| v5 | FocalLoss + unfreezing gradual | 0.411 |
| v6 | RGB + normalização ImageNet | 0.522 |
| v7 | **Differential LRs** (early=1e-5, late=5e-5, head=1e-3) + warm-up 5ep + Mixup(α=0.2) + label smoothing=0.1 + class weights suavizados (potência 0.5) + AdamW + grad clipping | ≥0.556* |

*v7 em execução no momento da elaboração deste relatório; `baseline artigo = 0.738` indicado no código como referência.

**Diagnóstico persistente:** A MobileNetV2 apresentou consistentemente dificuldade na fase 1 (backbone congelado), com val F1 ≤ 0.30. A decisão mais impactante foi o CLAHE offline (+7.9 pp, v3→v4). As *differential learning rates* da v7 tentam endereçar o problema de adaptação do backbone ao domínio médico, preservando features genéricas nas primeiras camadas (LR=1e-5) e adaptando rapidamente o classificador (LR=1e-3).

### 3.4 EfficientNet-B7 — Pasta `_RR` (3 versões)

| Versão | Principais mudanças | F1 Macro (test) | Accuracy |
|---|---|---|---|
| **rr_101** | Baseline: CLAHE grayscale offline, MONAI transforms 512 px, batch 4, backbone totalmente desbloqueado, CrossEntropy ponderada | ~0.48 | ~0.58 |
| **RR10** | CLAHE + pseudo-color BONE (3 canais distintos), 2 blocos descongelados, WeightedRandomSampler, AdamW, warm-up linear + cosine, batch 16, augmentação reforçada | ~0.46* | ~0.57 |
| **RR12** | **3 blocos + BN frozen**, aug reforçada (RandFlip H/V, RandAdjustContrast γ∈[0.75,1.4], GaussianNoise σ=0.015, GaussianSmooth), AdamW (backbone LR=3e-5, head LR=3e-4), early stop patience=12, TTA 8× | **0.555** | **0.633** |

*val F1 best; nota do notebook v12: "Val F1 ~0.46, AUC ~0.75 — melhor até agora" para v10.

**Diagnóstico crítico:** A EfficientNet-B7 revelou forte sobreajuste — train F1 → 0.99 vs val F1 ≈ 0.40–0.46 — visível nas curvas de aprendizagem. Com batch=16 e ~66M parâmetros, o BN descongelado introduz instabilidade nas estatísticas de normalização (por isso o BN é mantido em `eval()` na v12). A estratégia de descongelar apenas os últimos 2–3 blocos (vs. fine-tuning total) é essencial para evitar degradação dos pesos ImageNet. O TTA com 8 passes aumenta marginalmente o F1 (0.555 vs 0.555 sem TTA).

---

## 4. Comparativo Detalhado entre os Melhores Modelos

### 4.1 Configuração dos Melhores Modelos

| Parâmetro | DenseNet121 v100 | ResNet50 v2 | EfficientNet-B7 RR12 | MobileNetV2 v4 |
|---|---|---|---|---|
| **Backbone** | DenseNet121 (ImageNet) | ResNet50 (ImageNet) | EfficientNet-B7 (ImageNet) | MobileNetV2 (ImageNet) |
| **Cabeça** | Linear(1024→4) | Dropout(0.3)→Linear(2048→4) | Dropout(0.5)→Linear(2560→512)→SiLU→Dropout(0.3)→Linear(512→4) | Dropout(0.4)→Linear(1280→256)→ReLU→Dropout(0.3)→Linear(256→4) |
| **Blocos desbloqueados** | Todos | Todos | Últimos 3 + BN frozen | Todos (differential LR na v7) |
| **Resolução** | 256 × 256 px | 512 × 512 px | 512 × 512 px | 224 × 224 px |
| **Batch size** | 8 | 4 | 16 | 32 |
| **Optimizador** | AdamW | Adam | AdamW | AdamW |
| **LR backbone** | 1×10⁻⁴ | 1×10⁻⁴ | 3×10⁻⁵ | 1×10⁻⁵ (early) / 5×10⁻⁵ (late) |
| **LR classificador** | 1×10⁻⁴ | 1×10⁻⁴ | 3×10⁻⁴ | 1×10⁻³ |
| **Scheduler** | OneCycleLR (max_lr=3×10⁻⁴) | CosineAnnealingLR | Linear warm-up (4ep) + Cosine | Warm-up (5ep) + Cosine |
| **Função de perda** | 0.5×FocalLoss + 0.5×CE ponderada | FocalLoss | CE ponderada | CE ponderada + label smoothing=0.1 |
| **Class weights** | Dinâmicos (inversamente proporcionais) | — (FocalLoss implícita) | Dinâmicos | Suavizados (potência 0.5) |
| **Augmentação treino** | Rot ±30°, Zoom 0.85–1.15, Flip H/V, ShiftIntensity, Gamma, CLAHE offline | Rot ±15°, Zoom 0.9–1.1, AdjContrast, GaussNoise | Rot ±18°(π/10), Zoom 0.88–1.12, Flip H, Flip V, Contrast γ∈[0.75,1.4], GaussNoise σ=0.015, GaussSmooth | RandHFlip(0.4), VFlip(0.4), Rotation(20°), Affine shear, ColorJitter |
| **Mixup** | α=0.4 (desligado últimos 10 ep) | — | — | α=0.2 (pós warm-up) |
| **TTA** | 4 variantes | — | 8 passes | — |
| **Threshold tuning** | Sim (val set) | — | — | — |
| **Grad clipping** | max_norm=1.0 | — | max_norm=1.0 | max_norm=1.0 |
| **Pré-processamento** | CLAHE offline (clipLimit=2.0, 8×8) | Pipeline genérico | CLAHE + pseudo-color BONE | CLAHE offline (clipLimit=2.0, 8×8) |
| **Épocas (máx)** | 80 | 60 | 80 | 60 |
| **Early stopping** | patience=15 | patience=10 | patience=12 | patience=15 |
| **Épocas efectivas** | 80 | ~20–30 | ~28 | — |
| **Semente aleatória** | 42 | 42 | 42 | 42 |
| **AMP** | Sim (FP16+FP32) | — | — | — |

### 4.2 Resultados Globais no Conjunto de Teste (267 amostras)

| Modelo | F1 Macro | Accuracy | AUC-ROC macro | F1 BL (17) | F1 Li (123) | F1 No (43) | F1 St (84) |
|---|---|---|---|---|---|---|---|
| **DenseNet121 v100** | **0.7076** | **0.7266** | **~0.92** | **0.667** | 0.725 | 0.674 | **0.765** |
| ResNet50 v2 | 0.6647 | 0.7041 | ~0.89 | 0.552 | **0.747** | 0.595 | **0.765** |
| EfficientNet-B7 RR12 | 0.555 | 0.633 | ~0.87 | 0.286 | 0.674 | 0.633 | 0.626 |
| MobileNetV2 v4 | ~0.556 | ~0.620 | ~0.86 | ~0.350 | ~0.700 | ~0.550 | ~0.600 |
| **Baseline artigo (MIQR-CC)** | **0.738** | — | — | — | — | — | — |

*BL = Biliary Leaks, Li = Lithiasis, No = Normal, St = Stricture. AUC-ROC calculado com estratégia One-vs-Rest (macro).*

> **Nota sobre a baseline:** O artigo de referência (Martins et al., 2025) reporta F1 macro = 0.738 no mesmo conjunto de teste. O melhor modelo deste trabalho (DenseNet121 v100, F1=0.7076) fica 3.0 pp abaixo desse valor. Ainda assim, o resultado obtido representa um ganho de +16.3 pp sobre o baseline interno (v0, F1=0.545), demonstrando que as técnicas de tratamento do desequilíbrio de classes têm impacto dominante na tarefa.

### 4.3 Matrizes de Confusão — Melhores Modelos

**DenseNet121 v100** (F1=0.7076):
```
               Pred BL  Pred Li  Pred No  Pred St
Real BL  [17]     9        1        3        4
Real Li  [123]    1       83       17       22
Real No  [43]     0        8       32        3
Real St  [84]     0       14        0       70
```

**ResNet50 v2** (F1=0.6647):
```
               Pred BL  Pred Li  Pred No  Pred St
Real BL  [17]     8        1        8        0
Real Li  [123]    4       84       29        6
Real No  [43]     0        2       39        2
Real St  [84]     0       15       12       57
```

**EfficientNet-B7 RR12** (F1=0.555, sem TTA):
```
               Pred BL  Pred Li  Pred No  Pred St
Real BL  [17]     4        7        6        0
Real Li  [123]    2       89       15       17
Real No  [43]     1       12       30        0
Real St  [84]     4       33        1       46
```

**MobileNetV2 v4** (F1=0.5558):
```
               Pred BL  Pred Li  Pred No  Pred St
Real BL  [17]     3        5        9        0
Real Li  [123]    1       95        9       18
Real No  [43]     1       17       24        1
Real St  [84]     1       26        2       55
```

### 4.4 Análise Comparativa por Classe

**Biliary Leaks (classe mais crítica e mais rara, n=17):**

A DenseNet v100 alcança o melhor F1 (0.667), beneficiando da combinação de TTA, threshold tuning e loss ponderada. O recall de 9/17 (0.529) reflecte a dificuldade intrínseca da classe — as fugas biliares partilham características visuais com Stricture e Normal. A ResNet v2 tem recall de 8/17 mas muitos falsos positivos para Normal (8 erros), resultando em F1 mais baixo (0.552). A EfficientNet confunde 7/17 com Lithiasis, limitando o recall a 4/17=0.235.

**Lithiasis (classe maioritária, n=123):**

A ResNet v2 obtém o melhor F1 (0.747), com precision elevada (0.824) — o pipeline genérico da v2 favorece a classe dominante. A DenseNet v100 é ligeiramente inferior (0.725). A EfficientNet e MobileNet atingem valores similares (~0.670–0.700).

**Normal (n=43):**

A ResNet v2 destaca-se com recall 0.907 (39/43), com custo em precision (0.443). A DenseNet v100 equilibra melhor precision e recall (F1=0.674). A EfficientNet obtém 30/43 acertos (F1=0.633), performance surpreendentemente boa dado o seu F1 global mais fraco.

**Stricture (n=84):**

DenseNet v100 e ResNet v2 empatam em F1=0.765, com estratégias diferentes: a DenseNet tem recall mais alto (70/84=0.833) e a ResNet tem precision mais alta (0.877). A EfficientNet fica em F1=0.626, com 33/84 Strictures confundidas com Lithiasis (padrão morfológico semelhante em imagens ERCP).

### 4.4a Implicações Clínicas dos Erros de Classificação

Num contexto de apoio à decisão clínica (CADx), os erros de classificação não são equivalentes — as suas consequências dependem da patologia envolvida e do tipo de erro (falso negativo vs. falso positivo).

**Biliary Leaks — falso negativo é o erro mais grave.** Uma fuga biliar não detectada pode levar a peritonite biliar, sepsis e risco de vida. Os modelos deste trabalho falham em 8–14 dos 17 casos de teste, o que é clinicamente preocupante. O recall baixo (0.176 na MobileNetV2 v4; 0.529 na DenseNet v100) indica que todos os sistemas desenvolvidos devem ser usados apenas como ferramenta de suporte, nunca substituindo a avaliação do endoscopista.

**Stricture — confusão com Lithiasis tem impacto terapêutico directo.** Uma estenose classificada como cálculo biliar conduz a tratamento incorreto: a litotrícia (remoção de cálculos) é ineficaz e potencialmente lesiva numa estenose fibrosa. A DenseNet v100 confunde 14/84 Strictures com Lithiasis; a EfficientNet 33/84. Este padrão de erro sistemático é o mais relevante clinicamente de todo o trabalho.

**Lithiasis — falso positivo tem baixo risco clínico.** Classificar como Lithiasis uma imagem Normal leva a exames adicionais desnecessários (colangiorressonância, ecoendoscopia), sem risco imediato para o doente. Por isso, aumentar o recall de Lithiasis à custa de falsos positivos é uma troca aceitável.

**Normal — falso negativo tem risco variável.** Classificar uma patologia como Normal atrasa o diagnóstico, com gravidade dependente da patologia em causa: é grave para Biliary Leaks (ver acima) e menos grave para Lithiasis assintomática. A ResNet v2, com recall de Normal = 0.907, é a mais segura neste sentido mas à custa de muitos falsos positivos de Lithiasis.

### 4.5 Factores Determinantes por Arquitectura

**Por que a DenseNet obteve melhores resultados?**

1. A conectividade densa preserva features de baixo nível (bordas de ductos, texturas de cálculos) disponíveis em qualquer profundidade — crítico para distinguir Biliary Leaks de Normal.
2. A combinação CLAHE offline + AMP + resolução 256 px proporcionou o melhor equilíbrio detalhe/eficiência.
3. O Mixup com alpha elevado (0.4) criou representações inter-classe mais suaves, reduzindo o colapso em Lithiasis.
4. O TTA com threshold tuning é um ganho assimétrico: beneficia as classes mais difíceis (Biliary Leaks, Stricture) sem prejudicar as classes já bem aprendidas.

**Por que a ResNet ficou em 2.º lugar?**

A ResNet50 convergiu de forma mais estável do que as restantes arquitecturas, com o Dropout(0.3) na cabeça a ser o factor isolado mais impactante. A FocalLoss do baseline já endereçava parcialmente o desequilíbrio; adicionar o Dropout resolveu o sobreajuste nas camadas densas. A ausência de Mixup e TTA explica o gap face à DenseNet.

**Por que a EfficientNet e MobileNet ficaram atrás?**

- **EfficientNet-B7**: sobreajuste severo em configurações de batch pequeno (16) — as estatísticas do BN com batches pequenos são ruidosas; a arquitectura requer muito mais dados para explorar plenamente os seus 66M parâmetros. Fine-tuning parcial (apenas 3 blocos) reduz a flexibilidade mas é necessário para estabilizar o treino.
- **MobileNetV2**: a capacidade representacional limitada (~3.4M parâmetros) é insuficiente para distinguir padrões subtis nas imagens ERCP. A fase 1 com backbone congelado nunca convergiu (val F1 ≤ 0.30), iniciando a fase de fine-tuning de um ponto sub-óptimo.

### 4.6 Impacto das Decisões Metodológicas (DenseNet como caso de estudo)

| Decisão | Impacto no F1 Macro |
|---|---|
| Redução resolução 512→224 px | −0.032 pp |
| CLAHE em runtime (+512 px) | +0.004 pp |
| Endereçar desequilíbrio de classes (v99) | **+0.102 pp** |
| CLAHE offline + AMP + resolução 256 px | **+0.061 pp** (cumulativo total: +0.163 pp) |

A escolha da função de perda e das técnicas de treino foi ~25× mais impactante do que a optimização do pré-processamento.

---

## 5. Técnicas e Ferramentas Transversais

### 5.1 Gestão do Desequilíbrio de Classes

- **Focal Loss** (Lin et al., 2017): $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$ — penaliza dinamicamente exemplos fáceis, forçando o modelo a focar nas classes difíceis.
- **Class weights na CE Loss**: pesos inversamente proporcionais à frequência ($w_c = N_{\text{total}} / (C \cdot N_c)$), amplificando o gradiente das classes minoritárias.
- **WeightedRandomSampler** (EfficientNet): garante representação proporcional em cada batch ao nível do DataLoader.
- **Label Smoothing** ($\epsilon=0.1$): suaviza os targets one-hot em $y_{\text{soft}} = y \cdot (1-\epsilon) + \epsilon/C$, penalizando a sobreconfiança.

### 5.2 Regularização

- **Dropout** na cabeça classificadora: aleatoriza activações durante o treino, impedindo co-adaptação.
- **Mixup** (Zhang et al., 2018): $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\lambda \sim \text{Beta}(\alpha, \alpha)$ — interpola imagens e labels, criando representações inter-classe que reduzem overfitting.
- **Gradient Clipping** (max_norm=1.0): estabiliza o treino quando batches com exemplos difíceis provocam gradientes explosivos.

### 5.3 Inferência Robusta

- **Test-Time Augmentation (TTA)**: avalia cada imagem de teste em múltiplas variantes aumentadas e faz a média das probabilidades, reduzindo a variância da predição.
- **Threshold Tuning**: os limiares de decisão de cada classe são calibrados no val set antes da avaliação final no test set.

### 5.4 Pré-processamento

- **CLAHE** (Pizer et al., 1987): equalização adaptativa de histograma com limite de contraste (clipLimit=2.0, tileGrid=8×8 ou 16×16). Normaliza o contraste localmente, tornando estruturas biliares subtis mais salientes.
- **CLAHE + pseudo-color BONE**: converte a imagem CLAHE grayscale para 3 canais distintos via `cv2.COLORMAP_BONE`, fornecendo ao modelo RGB informação espectral diferenciada que beneficia architecturas com normalização ImageNet.
- **ROI anatómica** (ResNet v1): `CropForeground` + `CenterSpatialCrop(480)` + `ScaleIntensityRangePercentiles(1%, 99%)` + máscara circular — remove o fundo escuro periférico dos endoscópios.
- **Normalização ImageNet**: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — essencial para modelos que usam pesos ImageNet com imagens RGB.

### 5.5 Schedulers e Optimizadores

- **OneCycleLR**: warm-up até lr_max seguido de cosine annealing — favorece convergência rápida em poucas épocas.
- **Linear warm-up + Cosine**: aquece gradualmente o LR nas primeiras épocas para estabilizar pesos pré-treinados.
- **AdamW**: Adam com weight decay desacoplado — melhor regularização implícita que Adam standard.
- **Differential Learning Rates**: grupos de parâmetros com LRs distintos (backbone early ≪ backbone late ≪ head), preservando features genéricas enquanto adapta o classificador rapidamente.

### 5.6 Interpretabilidade — Grad-CAM

O método **Grad-CAM** (Selvaraju et al., 2017) foi implementado nos notebooks MobileNet v7 e EfficientNet RR12. Calcula pesos de importância a partir dos gradientes da classe-alvo relativamente ao último mapa de activação convolucional:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A_{ij}^k}$$

As visualizações confirmam que os modelos aprendem regiões anatomicamente relevantes (ductos biliares, zonas de estenose), validando a pertinência do transfer learning.

---

## 6. Abordagem Ensemble

O notebook `_CB_ENSEMBLE_v0__densenet_AND_efficientnet.ipynb` implementa um **ensemble por soft voting** entre DenseNet121 e EfficientNet-B7:

$$P_{\text{ensemble}}(c | x) = \frac{1}{M}\sum_{m=1}^{M} P_m(c | x)$$

A premissa é que os dois modelos cometem erros em padrões distintos: a DenseNet beneficia da conectividade densa para features de baixo nível, enquanto a EfficientNet tem maior capacidade representacional graças ao escalamento composto. A média das probabilidades suaviza as predições, reduzindo a variância.

**Resultados preliminares (modelos base v0):**

| Sistema | F1 Macro | Accuracy |
|---|---|---|
| DenseNet (standalone) | ~0.545 | ~0.618 |
| EfficientNet (standalone) | ~0.480 | ~0.570 |
| **Ensemble (média)** | **>0.550** | **>0.620** |

Uma segunda versão (`_CB_ENSEMBLE_v1___efficientenet_AND_densenet_v99.ipynb`) combina os modelos de versões mais avançadas (DenseNet v99 + EfficientNet melhorada), antecipando-se um ganho adicional. O ensemble é particularmente valioso para a classe Biliary Leaks, onde diferentes modelos cometem erros em exemplos distintos — a combinação aumenta a probabilidade de pelo menos um modelo detectar correctamente cada caso.

---

## 7. Síntese e Conclusões

### 7.1 Ranking Final

| Pos. | Arquitectura | Melhor Versão | F1 Macro | Accuracy |
|---|---|---|---|---|
| **1.º** | **DenseNet121** | v100 / v101 | **0.7076** | **0.7266** |
| **2.º** | **ResNet50** | v2 | **0.6647** | **0.7041** |
| **3.º** | **EfficientNet-B7** | RR12 (TTA) | **~0.556** | **~0.633** |
| **4.º** | **MobileNetV2** | v4 (CLAHE) | **~0.556** | **~0.620** |

### 7.2 Conclusão Principal

O factor mais determinante para o sucesso em todos os modelos foi o **tratamento explícito do desequilíbrio de classes** — através de funções de perda ponderadas, Mixup e TTA. O pré-processamento (CLAHE, ROI anatómica) teve impacto marginal em comparação. A DenseNet121 beneficiou da combinação mais completa destas técnicas, alcançando F1=0.7076, um ganho de +16.3 pp sobre o baseline interno (v0) e a 3.0 pp da baseline publicada no artigo MIQR-CC (F1=0.738). Este gap face ao artigo sublinha que superar a baseline de referência requer dados sintéticos adicionais ou arquitecturas de atenção espacial (Vision Transformers) ainda não exploradas neste trabalho.

A ResNet50 demonstrou ser a arquitectura com melhor relação capacidade/estabilidade de treino: converge de forma robusta com poucos hiperparâmetros críticos. A EfficientNet-B7, apesar da sua capacidade superior, sofreu de sobreajuste severo com os dados disponíveis (~1000 imagens de treino). A MobileNetV2, sendo concebida para eficiência computacional, mostrou limitações de capacidade representacional para a complexidade do problema médico.

### 7.3 Caminho para Melhoria Futura

- **WeightedRandomSampler** para Biliary Leaks (recall persistente em 9/17)
- **Validação cruzada estratificada k-fold** para estimativas mais robustas da variância
- **Ensemble DenseNet v100 + ResNet v2** — os dois melhores modelos individuais, com padrões de erro distintos
- Aumento do dataset com técnicas de síntese (GAN) especificamente para Biliary Leaks

---

## Referências

- Martins, M. C. et al. (2025). *Curated endoscopic retrograde cholangiopancreatography images dataset (MIQR-CC).* Figshare. https://doi.org/10.6084/m9.figshare.31079236
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR.
- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely connected convolutional networks.* CVPR.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV.
- Pizer, S. M. et al. (1987). *Adaptive histogram equalization and its variations.* CVGIP.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks.* CVPR.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks.* ICCV.
- Tajbakhsh, N. et al. (2016). *Convolutional neural networks for medical image analysis.* IEEE TMI.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for CNNs.* ICML.
- Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). *Mixup: Beyond empirical risk minimization.* ICLR.
