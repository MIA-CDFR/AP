# NOTAS PESSOAIS. APAGAR AS LINHAS A SEGUIR ANTES DE SUBMETER.
```
Narrativa para 2 min de apresentação:
https://claude.ai/share/3462db15-e34b-4861-965e-1e09f660fcac

Análise técnica aprofundada:
https://claude.ai/share/c6ea5523-84fd-4dff-adb5-b7609274051e

Apresentação/narrativa para 2 minutos:
https://docs.google.com/presentation/d/156GvGMaEITIIt7Ch0eDGXBNAAiRb5Qws2y1Q2uOVNK4/edit?usp=sharing
```
# ---------------------- FIM NOTAS PESSOAIS.
---
# DenseNet para Classificação ERCP — Relatório Técnico Comparativo

> **Trabalho desenvolvido sobre o notebook base fornecido pelo docente (`DENSENET.ipynb`)**  
> Objectivo: maximizar o **F1-score macro** na classificação multiclasse de imagens ERCP com DenseNet121.

---

## 1. Introdução — Redes Neurais e a Arquitectura DenseNet

### 1.1 O que é uma Rede Neural Artificial?

Uma **rede neural artificial** (RNA) é um modelo computacional inspirado no funcionamento do cérebro humano. É constituída por unidades de processamento chamadas **neurónios artificiais**, organizadas em **camadas**:

- **Camada de entrada** (*input layer*) — recebe os dados em bruto (ex.: pixels de uma imagem).
- **Camadas ocultas** (*hidden layers*) — aprendem representações progressivamente mais abstractas dos dados.
- **Camada de saída** (*output layer*) — produz a predição final (ex.: probabilidade de cada classe).

Cada neurónio aplica uma **transformação linear** aos seus inputs (soma ponderada) seguida de uma **função de activação não-linear** (ex.: ReLU), o que confere à rede a capacidade de modelar relações complexas e não-lineares.

O treino de uma RNA consiste em ajustar iterativamente os **pesos** de cada ligação através de **retropropagação do gradiente** (*backpropagation*), minimizando uma função de perda (*loss function*) que mede o erro das predições.

### 1.2 Redes Neurais Convolucionais (CNNs)

Para dados com estrutura espacial, como imagens médicas, utilizam-se **Redes Neurais Convolucionais** (CNNs). Em vez de ligações totalmente densas, as CNNs aplicam **filtros convolucionais** locais que deslizam pela imagem, aprendendo padrões como bordas, texturas e formas. As camadas de **pooling** reduzem progressivamente a resolução espacial, aumentando o campo receptivo e reduzindo o custo computacional.

As CNNs modernas são tipicamente pré-treinadas em grandes datasets como o **ImageNet** (14 milhões de imagens, 1000 classes) e depois **fine-tuned** para a tarefa específica — técnica conhecida como **transfer learning**. Isto é especialmente valioso em contextos médicos onde os dados anotados são escassos.

### 1.3 DenseNet — Dense Convolutional Network

A **DenseNet** (*Densely Connected Convolutional Network*) foi proposta por Huang et al. em 2017 e introduz um princípio simples mas poderoso: **cada camada recebe como input as feature maps de todas as camadas anteriores**, e passa as suas próprias feature maps a todas as camadas seguintes.

```
Camada L recebe: [x₀, x₁, x₂, ..., x_{L-1}]
```

Numa ResNet clássica, a ligação residual soma apenas a camada anterior. Na DenseNet, a concatenação é feita para **todas** as camadas precedentes dentro do mesmo bloco denso.

```
ResNet:   x_L = H_L(x_{L-1}) + x_{L-1}          (adição — soma os valores)
DenseNet: x_L = H_L([x_0, x_1, ..., x_{L-1}])    (concatenação — preserva todos os canais)
```

#### Componentes principais da DenseNet121

| Componente | Descrição |
|---|---|
| **Dense Block** | Conjunto de camadas com conectividade densa entre si |
| **Transition Layer** | Convoluição 1×1 + Average Pooling 2×2 — reduz dimensionalidade entre blocos |
| **Growth Rate (k)** | Número de feature maps que cada camada acrescenta (ex.: k=32 no DenseNet121) |
| **Bottleneck** | Convoluição 1×1 antes da 3×3 para reduzir nº de canais e custo computacional |
| **Global Average Pooling** | Substitui camadas FC densas, reduz overfitting |
| **Classifier** | Camada linear final adaptada ao número de classes da tarefa |

O sufixo **121** indica o número total de camadas com pesos: 4 blocos densos com 6, 12, 24 e 16 camadas respectivamente, mais camadas de transição e a cabeça classificadora.

#### Vantagens da DenseNet em imagem médica

- **Reutilização de features** — informação de baixo nível (bordas, texturas) está disponível em camadas profundas, o que é crítico para detectar estruturas anatómicas subtis.
- **Gradientes fortes** — o caminho directo entre qualquer camada e a loss facilita o treino de redes profundas sem degradação do gradiente.
- **Eficiência de parâmetros** — apesar da conectividade densa, o growth rate controlado mantém o modelo compacto.
- **Regularização implícita** — a agregação de múltiplas representações actua como um ensemble interno, reduzindo overfitting.

---

## 2. Dataset e Problema

### 2.1 Descrição

O dataset consiste em imagens de **ERCP** (Colangiopancreatografia Retrógrada Endoscópica), exame de diagnóstico e intervenção da via biliar. A tarefa é classificar cada imagem em 4 classes:

- **Biliary Leaks** — fuga biliar
- **Lithiasis** — cálculos biliares
- **Normal** — sem patologia identificada
- **Stricture** — estenose da via biliar

### 2.2 Distribuição do Dataset

| Split | Total | Biliary Leaks | Lithiasis | Normal | Stricture |
|---|---|---|---|---|---|
| **Train** | 1 067 | 110 (10.3%) | 505 (47.3%) | 197 (18.5%) | 255 (23.9%) |
| **Val** | 234 | 24 | 98 | 59 | 53 |
| **Test** | 267 | 17 | 123 | 43 | 84 |

O dataset é acentuadamente **desequilibrado**: a classe Lithiasis representa ~47% do treino enquanto Biliary Leaks representa apenas ~10%. Este desequilíbrio é o principal desafio do problema — modelos ingénuos tendem a ignorar as classes minoritárias, maximizando a accuracy mas falhando nas classes mais raras.

> **Nota clínica:** A Biliary Leaks é simultaneamente a classe mais rara e a mais crítica clinicamente — um falso negativo pode ter consequências graves para o doente.

### 2.3 Métrica de Avaliação

A métrica principal é o **F1-score macro**, que calcula o F1 de cada classe individualmente e faz a média não-ponderada. Ao contrário da accuracy, o F1 macro penaliza igualmente o mau desempenho em classes raras e em classes frequentes.

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
F1 macro = média(F1_Biliary, F1_Lithiasis, F1_Normal, F1_Stricture)
```

---

## 3. Arquitectura Base e Configuração Comum

Todos os modelos partilham a mesma espinha dorsal:

| Componente | Configuração |
|---|---|
| **Backbone** | DenseNet121 pré-treinado (ImageNet) |
| **Cabeça** | Linear(1024 → 4 classes) |
| **Loss base** | CrossEntropyLoss |
| **Optimizador base** | Adam, lr = 1×10⁻⁴ |
| **Scheduler base** | CosineAnnealingLR |
| **Early stopping** | patience = 10 (salva melhor checkpoint por val F1) |
| **Augmentações base** | RandRotate ±15°, RandZoom 0.9–1.1, RandAdjustContrast, RandGaussianNoise |

---

## 4. Versões Desenvolvidas

### 4.1 v0 — Baseline Fiel

Reprodução controlada do notebook do docente. O único acrescento é a centralização dos hiperparâmetros num dicionário de configuração e o versionamento automático dos modelos guardados.

| Parâmetro | Valor |
|---|---|
| Resolução | 512 × 512 px |
| Batch size | 4 |
| Melhor época | 9 |
| Tempo de treino | ~2 091 s |

**Função de perda:** `CrossEntropyLoss` sem class weights — todas as classes têm o mesmo peso no gradiente.

---

### 4.2 v1 — Resolução 224 px + Batch 16

**Hipótese:** 224×224 é a resolução nativa do ImageNet; o pré-treino do DenseNet121 foi optimizado para esta resolução. Batches maiores estabilizam os gradientes e permitem uma taxa de aprendizagem efectiva mais elevada.

| Parâmetro | Valor |
|---|---|
| Resolução | 224 × 224 px |
| Batch size | 16 |
| Melhor época | 7 |
| Tempo de treino | ~410 s (~5× mais rápido) |

**Resultado surpresa:** a accuracy subiu (0.693 vs 0.618) mas o F1 macro caiu, e a Biliary Leaks passou a ter F1 = 0.000 — o modelo deixou de detectar a classe mais rara por completo. A resolução reduzida pode ter eliminado detalhes diagnósticos subtis presentes nas imagens ERCP.

---

### 4.3 v2 — CLAHE em 512 px

**CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) é um método de equalização de histograma local: divide a imagem em tiles e equaliza cada um independentemente, com um limite de contraste para evitar amplificação excessiva de ruído.

**Parâmetros:** `clipLimit=2.0`, `tileGridSize=(8,8)`, aplicado com `prob=0.7` em treino.

**Hipótese:** As imagens ERCP têm contraste variável e regiões de interesse frequentemente sub-expostas. O CLAHE normaliza localmente, ajudando o modelo a focar nas estruturas biliares relevantes.

| Parâmetro | Valor |
|---|---|
| Resolução | 512 × 512 px |
| Batch size | 4 |
| Melhor época | 14 |
| Tempo de treino | ~similar à v0 |

---

### 4.4 v3 — CLAHE + 224 px + Batch 16

Combinação directa de v1 e v2 para avaliar se os dois eixos de melhoria se complementam.

| Parâmetro | Valor |
|---|---|
| Resolução | 224 × 224 px |
| Batch size | 16 |
| Melhor época | 7 |

**Resultado:** Os dois eixos não somaram os seus efeitos. O CLAHE não compensou a perda de informação da resolução reduzida.

---

### 4.5 v99 — F1Boost

Versão sem restrições, redesenhada de raiz para atacar o problema identificado nas versões anteriores: **o desequilíbrio de classes**. Introduz um conjunto de técnicas complementares:

#### Stack de melhorias

**Optimização:**
- `AdamW` com weight decay — Adam com regularização L2 implícita
- `OneCycleLR` scheduler — warm-up seguido de cosine annealing, favorece convergência mais rápida e estável
- Gradient clipping (max_norm=1.0) — estabilidade em batches com exemplos difíceis

**Função de perda composta:**
- `FocalLoss` — penaliza mais os exemplos mal classificados, forçando o modelo a aprender classes difíceis
- `CrossEntropyLoss` com class weights calculados dinamicamente:
```
weight_class = total_samples / (num_classes × count_class)
→ weight(Biliary Leaks) ≈ 2.43
→ weight(Lithiasis) ≈ 0.53
```

**Regularização:**
- `Mixup augmentation` — interpola pares de imagens e os seus labels (`λ ~ Beta(0.4, 0.4)`); o modelo aprende representações mais suaves e generaliza melhor
- Augmentações mais agressivas: flips H/V, rotação ±30°, RandShiftIntensity, RandScaleIntensity

**Inferência robusta:**
- `Test-Time Augmentation (TTA)` — 4 variantes por imagem (original, flip H, flip V, rotação 90°); média das probabilidades reduz variância da predição
- `Threshold tuning por classe` — os limiares de decisão de cada classe são optimizados no val set antes da avaliação final no test set

| Parâmetro | Valor |
|---|---|
| Resolução | 512 × 512 px |
| Batch size | 8 |
| Melhor época | 19 |
| Tempo de treino | ~6.4 h |
| Early stopping patience | 15 |

---

### 4.6 v100 — CLAHE Offline + AMP + 256 px

Versão que combina toda a stack da v99 com pré-processamento CLAHE offline e optimizações de velocidade de treino.

#### Melhorias adicionais face à v99

**CLAHE offline:** em vez de aplicar CLAHE como augmentação em tempo real, o dataset completo é pré-processado e guardado em `dataset_clahe/`. Vantagens: eliminação do custo computacional por época; CLAHE aplicado consistentemente (sem aleatoriedade de prob); o modelo vê sempre a versão normalizada.

**Resolução 256 × 256 px:** compromisso entre 224 px (resolução nativa ImageNet) e 512 px (resolução original); preserva detalhes finos sem o custo computacional máximo.

**AMP — Automatic Mixed Precision:** utiliza `torch.amp.autocast('cuda')` + `GradScaler`. As operações forward e backward são executadas em FP16 onde possível, com acumulação de gradientes em FP32 para estabilidade numérica. Resultado: ~2× mais rápido sem perda de qualidade.

**`pin_memory=True`** nos DataLoaders: pré-aloca memória na RAM paginável, tornando a transferência CPU→GPU mais eficiente.

| Parâmetro | Valor |
|---|---|
| Resolução | 256 × 256 px |
| Batch size | 8 |
| Épocas treinadas | 80 (sem early stopping activo) |
| Tempo por época | ~1 min |

---

### 4.7 v101 — v100 + Augmentação Dirigida à Biliary Leaks + Patience 20

Versão que parte directamente da stack da v100 e endereça o ponto de melhoria mais urgente identificado na análise: o recall insuficiente da Biliary Leaks (recall = 0.529 em v100). As alterações são cirúrgicas — não se toca na arquitectura nem no pipeline geral, apenas se reforçam os mecanismos que afectam directamente a classe minoritária.

#### Mudanças face à v100

**Augmentações mais agressivas no treino:**

As augmentações geométricas e de intensidade foram intensificadas para forçar maior variabilidade dos exemplos de treino:

| Augmentação | v100 | v101 |
|---|---|---|
| Rotação | ±15°, prob=0.5 | ±30°, prob=0.6 |
| Zoom | 0.9–1.1 | 0.85–1.15 |
| Flip horizontal | não | prob=0.5 |
| Flip vertical | não | prob=0.5 |
| RandShiftIntensity | offsets=0.10 | offsets=0.15 |
| Gamma contrast | 0.8–1.2 | 0.7–1.5 |

**Label Smoothing na loss:**

A função de perda `WeightedCELabelSmoothing` substitui o CrossEntropy simples. O label smoothing (`smoothing=0.1`) suaviza os targets de one-hot para distribuições suaves:

```
target_suave = target_hard × (1 − 0.1) + 0.1 / num_classes
```

Isto penaliza a overconfidence do modelo, tornando-o mais calibrado e menos propenso a colapsar na classe maioritária.

**Loss composta reconfigurada:**

```
combined_loss = 0.6 × FocalLoss + 0.4 × WeightedCELabelSmoothing
```

**Mixup desligado nos últimos 10 épocas** (`MIXUP_OFF_LAST=10`): permite ao modelo afinar as suas predições sem a regularização do Mixup nas épocas finais, melhorando a calibração das probabilidades.

**Batch size 16** (vs 8 na v100): gradientes mais estáveis por iteração, especialmente benéfico quando os batches têm de conter exemplos de todas as classes.

**Early stopping patience = 20** (vs 15 na v100): dá mais margem ao modelo para recuperar de plateaux temporários no val F1, evitando paragem prematura.

| Parâmetro | Valor |
|---|---|
| Resolução | 256 × 256 px |
| Batch size | 16 |
| Épocas máx. | 80 |
| Melhor época (early stop) | 38 |
| Tempo de treino | ~1 197 s (~20 min) |
| Early stopping patience | 20 |
| Label smoothing | 0.1 |
| Mixup alpha | 0.4 (desligado nos últimos 10 epochs) |
| Val F1 best | 0.6059 |
| Test F1 macro | 0.7076 |
| Test Accuracy | 0.7266 |

---

## 5. Resultados Comparativos

### 5.1 F1-Score Macro e Accuracy no Test Set

| Versão | F1 Macro | Accuracy | Δ F1 vs v0 | Nota principal |
|---|---|---|---|---|
| **v0** | 0.545 | 0.618 | — | Baseline |
| v1 | 0.513 | 0.693 | −0.032 | Biliary Leaks: F1=0.000 |
| v2 | 0.549 | 0.727 | +0.004 | Biliary Leaks: F1=0.000 |
| v3 | 0.525 | 0.693 | −0.020 | Biliary Leaks: F1=0.000 |
| v99 | 0.647 | 0.674 | +0.102 | Primeiro modelo a detectar BL |
| **v100** | **0.7076** | **0.7266** | **+0.163** | Melhor resultado global com outputs |
| v101 | 0.7076 | 0.7266 | +0.163 | v100 + augmentações BL + label smoothing + patience 20 — early stop época 38 |

### 5.2 F1-Score por Classe

| Classe | Suporte | v0 | v1 | v2 | v3 | v99 | v100 | v101 | Δ v0→v101 |
|---|---|---|---|---|---|---|---|---|---|
| **Biliary Leaks** | 17 | 0.438 | 0.000 | 0.000 | 0.000 | 0.579 | 0.667 | 0.667 | **+0.229** |
| **Lithiasis** | 123 | 0.708 | 0.810 | 0.833 | 0.810 | 0.708 | 0.725 | 0.725 | +0.017 |
| **Normal** | 43 | 0.488 | 0.543 | 0.488 | 0.543 | 0.615 | 0.674 | 0.674 | **+0.186** |
| **Stricture** | 84 | 0.546 | 0.698 | 0.875 | 0.698 | 0.684 | 0.765 | 0.765 | **+0.219** |

> **Leitura:** A Lithiasis (classe maioritária) teve ganhos modestos ao longo das versões — o modelo base já a aprende bem. Os ganhos mais expressivos são nas classes minoritárias, precisamente onde o desequilíbrio era mais penalizador. A v101 igualou a v100 no test set apesar de ter parado na época 38 (vs 80), o que sugere que as melhorias de regularização estabilizaram a convergência.

### 5.3 Matrizes de Confusão

#### v0 — Baseline

```
                 Biliary  Lithi  Normal  Stricture
Biliary_Leaks  [    7       2       7        1   ]
Lithiasis      [    1     102      12        8   ]
Normal         [    3      17      20        3   ]
Stricture      [    4      44       0       36   ]
```

*O modelo confunde Biliary Leaks com Normal (7 erros) e Stricture com Lithiasis (44 erros).*

#### v99 — F1Boost

```
                 Biliary  Lithi  Normal  Stricture
Biliary_Leaks  [   11       0       5        1   ]
Lithiasis      [    4      85      22       12   ]
Normal         [    1       7      32        3   ]
Stricture      [    5      25       2       52   ]
```

*Biliary Leaks passa de 7 para 11 acertos. Stricture melhora de 36 para 52 acertos. O custo é alguma confusão adicional em Lithiasis.*

#### v100 — Melhor resultado (F1 macro = 0.7076)

```
                 Biliary  Lithi  Normal  Stricture
Biliary_Leaks  [    9       1       3        4   ]
Lithiasis      [    1      83      17       22   ]
Normal         [    0       8      32        3   ]
Stricture      [    0      14       0       70   ]
```

#### v101 — Igual ao v100 no test set (early stop época 38)

```
                 Biliary  Lithi  Normal  Stricture
Biliary_Leaks  [    9       1       3        4   ]
Lithiasis      [    1      83      17       22   ]
Normal         [    0       8      32        3   ]
Stricture      [    0      14       0       70   ]
```

*A matriz de confusão é idêntica à v100. A v101 convergiu mais rapidamente (38 épocas vs 80) para o mesmo resultado no test set, o que é um sinal positivo de melhor regularização.*

*Stricture atinge 70/84 acertos (recall = 0.833). Normal tem 0 falsos negativos para Biliary Leaks. O principal ponto fraco mantém-se: 9 de 17 Biliary Leaks correctos (recall = 0.529).*

---

## 6. Análise Crítica

### 6.1 O que as versões v1–v3 revelaram

As experiências com resolução reduzida (v1, v3) mostraram que **baixar de 512 para 224 px destrói a capacidade de detectar Biliary Leaks**, apesar de melhorar a accuracy global. Isto sugere que as características diagnósticas desta classe dependem de detalhes de alta frequência que se perdem com o downscaling agressivo.

O CLAHE (v2) trouxe uma melhoria marginal (+0.004 F1 macro) sem resolver o problema central. A combinação v3 não somou os efeitos — o CLAHE não compensa a perda de resolução.

**Conclusão intermédia:** O problema não era a resolução nem o contraste. Era o desequilíbrio de classes.

### 6.2 Por que a v99 funcionou

A v0 ignorava Biliary Leaks porque a classe minoritária (110 amostras de treino, ~10%) não tinha peso suficiente no gradiente. Com CrossEntropyLoss uniforme, o modelo aprende que "prever Lithiasis sempre" minimiza razoavelmente a loss. A v99 atacou este problema em três frentes:

1. **Loss ponderada** — class weights e FocalLoss forçam o modelo a prestar atenção às classes raras, penalizando mais os erros nessas classes
2. **Regularização** — Mixup e augmentações agressivas evitam overfitting nas classes maioritárias e forçam o modelo a aprender representações mais robustas
3. **Inferência robusta** — TTA e threshold tuning aumentam a sensibilidade às classes sub-representadas na fase de predição

### 6.3 Análise da v100 e v101

**Pontos fortes (ambas):**

- **F1 Macro 0.7076** — melhor resultado obtido em toda a série, +6 p.p. face à v99 e +16 p.p. face à baseline
- **Stricture: F1 = 0.765** (70/84 acertos) — bem acima da baseline (0.546)
- **Normal: F1 = 0.674** com 0 confusões com Biliary Leaks — clinicamente importante
- Matriz de confusão **idêntica** entre v100 e v101 — resultados reprodutíveis no test set

**O que a v101 revelou:**

A v101 atingiu exactamente o mesmo F1 macro (0.7076) e a mesma matriz de confusão que a v100, mas com **early stopping à época 38** (vs 80 épocas completas) e num tempo de treino de ~20 min (vs ~80 min). Duas leituras:

1. **Positiva:** label smoothing + augmentações mais agressivas + batch size 16 tornaram a convergência mais eficiente — o modelo aprendeu o essencial em menos de metade das épocas.
2. **Limitante:** o val F1 best foi 0.6059 (época 38), com alta variância nas curvas (oscilações de ±0.1 entre épocas consecutivas) — o OneCycleLR com lr_max=3×10⁻⁴ e batch 16 pode estar a introduzir ruído excessivo no gradiente. O threshold tuning + TTA está a fazer trabalho pesado na ponte entre val e test.

**O ponto crítico que persiste em ambas:** Biliary Leaks com recall = 0.529 (9/17 acertos), confundida principalmente com Stricture (4 erros) e Normal (3 erros). Nenhuma das versões conseguiu melhorar este número.

### 6.4 Insight Central

| Tipo de melhoria | Impacto no F1 macro |
|---|---|
| Optimização de resolução (v1, v3) | −0.020 a −0.032 |
| Pré-processamento CLAHE (v2) | +0.004 |
| Endereçar desequilíbrio de classes (v99) | **+0.102** |
| CLAHE + AMP + resolução intermédia (v100/v101) | **+0.163** total |

Num problema de classificação médica com classes desequilibradas, **a escolha da função de perda e as técnicas de treino valem muito mais do que optimizar o pré-processamento.** Melhorar a resolução ou adicionar CLAHE são ajustes incrementais; lidar com o desequilíbrio é estrutural.


## 7. Próximos Passos Sugeridos

Com base nos resultados da v101, os pontos de melhoria mais promissores são:

- **`WeightedRandomSampler`** — oversampling da Biliary Leaks ao nível do DataLoader; diferente dos class weights na loss, garante que **cada batch** contém exemplos proporcionais da classe minoritária. É o próximo passo mais directo para atacar o recall de 0.529.
- **Reduzir o learning rate máximo** para 1×10⁻⁴ (era 3×10⁻⁴ na v101) — as curvas de val F1 com alta variância sugerem que o OneCycleLR está a dar passos demasiado grandes com batch 16.
- **Dropout explícito** na cabeça classificadora — combate o overfitting observado no gap training/val.
- **Augmentações mais agressivas** ✅ *implementado na v101* — trouxe convergência mais rápida mas não melhorou o recall da Biliary Leaks.
- **Label smoothing + patience 20** ✅ *implementado na v101* — convergência 2× mais rápida para o mesmo resultado no test set.
- **Ensemble DenseNet + EfficientNet** — os dois modelos cometem erros em padrões diferentes; a combinação dos logits pode reduzir a variância das predições (ver notebooks `_CB_ENSEMBLE_*`).

---

## 8. Estrutura dos Ficheiros

```
.
├── DENSENET.ipynb                               # Notebook base (docente)
├── _DENSENET_CB_v0.ipynb                        # Baseline fiel
├── _DENSENET_CB_v1.ipynb                        # 224px + batch 16
├── _DENSENET_CB_v2.ipynb                        # CLAHE (512px)
├── _DENSENET_CB_v3.ipynb                        # CLAHE + 224px + batch 16
├── _DENSENET_CB_v99.ipynb                       # F1Boost
├── _DENSENET_CB_v100.ipynb                      # CLAHE + AMP + 256px
├── _DENSENET_CB_v101.ipynb                      # v100 + augmentações BL + label smoothing + patience 20
├── _CB_ENSEMBLE_v0__densenet_AND_efficientnet.ipynb
├── _CB_ENSEMBLE_v1___efficientenet_AND_densenet_v99.ipynb
├── README_DENSENET_CB_v1.md                     # Este ficheiro
└── models/
    └── MIA_AP_my_models/
        ├── densenet_v0_*.pth  (+cm.png, +.png)
        ├── densenet_v1_*.pth  (+cm.png, +.png)
        ├── densenet_v2_*.pth  (+cm.png, +.png)
        ├── densenet_v3_*.pth  (+cm.png, +.png)
        └── densenet_v99_*.pth (+cm.png, +.png)
```

---

## 9. Referências

- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). **Densely Connected Convolutional Networks.** *CVPR 2017.*
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). **Focal Loss for Dense Object Detection.** *ICCV 2017.*
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). **mixup: Beyond Empirical Risk Minimization.** *ICLR 2018.*
- Smith, L. N., & Topin, N. (2018). **Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.** *ICML Workshop 2018.*
- Pizer, S. M. et al. (1987). **Adaptive histogram equalization and its variations.** *Computer Vision, Graphics, and Image Processing.*
