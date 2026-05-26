# Evolução do Modelo DenseNet — v0 até v100

## Contexto

Este documento resume a evolução do modelo DenseNet utilizado para classificação automática de imagens ERCP em quatro classes clínicas:

- Biliary_Leaks
- Lithiasis
- Normal
- Stricture

A evolução foi construída de forma incremental, testando alterações em:
- resolução das imagens;
- batch size;
- preprocessamento;
- data augmentation;
- loss functions;
- schedulers;
- regularização;
- estratégias de inferência.

> Nota: apenas foram disponibilizadas as versões v0, v1, v2, v3, v99 e v100.  
> As versões intermédias (v4 → v98) não estavam presentes nos notebooks enviados.

---

# v0 — Baseline inicial

## Objetivo
Criar a primeira baseline funcional utilizando DenseNet para classificação multi-classe.

## Configuração principal
- Input size: 512×512
- Batch size: 4
- Epochs: 60
- Learning rate: 1e-4

## Características
- Pipeline base MONAI
- Augmentações simples
- Scheduler básico
- Sem otimizações avançadas
- Sem técnicas específicas para imbalance

## Resultado
A versão v0 serviu como referência inicial para todas as experiências posteriores.

### F1-score macro
**≈ 0.5466**

---

# v1 — Otimização de dimensão e batch

## Alterações introduzidas
- Redução da resolução:
  - 512×512 → 224×224
- Aumento do batch size:
  - 4 → 16

## Objetivo
Melhorar:
- velocidade de treino;
- estabilidade dos gradientes;
- utilização de memória;
- generalização do modelo.

## Impacto esperado
- treino mais rápido;
- maior estabilidade;
- menor overfitting.

### F1-score macro
**≈ 0.5910**

---

# v2 — Introdução de CLAHE

## Alterações introduzidas
Foi introduzido:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

## Objetivo
Melhorar:
- contraste local;
- visibilidade de estruturas clínicas;
- robustez em imagens médicas.

## Configuração CLAHE
- clipLimit = 2.0
- tileGridSize = (8,8)
- probabilidade = 0.7

## Resultado
Esta versão apresentou uma melhoria significativa do desempenho global.

### F1-score macro
**≈ 0.6385**

---

# v3 — CLAHE + otimização de resolução

## Alterações introduzidas
Combinação de:
- CLAHE da v2
- resolução reduzida para 224×224
- batch size aumentado para 16

## Objetivo
Combinar:
- melhor contraste;
- treino mais rápido;
- melhor estabilidade.

## Observações
Apesar da melhoria computacional, o ganho de F1 não foi superior ao da v2.

### F1-score macro
**≈ 0.5515**

---

# v99 — F1Boost

## Objetivo
Criar uma versão fortemente otimizada para maximizar F1-score macro.

## Principais melhorias

### Otimizador e Scheduler
- AdamW
- OneCycleLR

## Loss Functions
Combinação de:
- Focal Loss
- CrossEntropy
- class weights
- label smoothing

## Regularização
- Mixup augmentation

## Data Augmentation
Augmentações mais agressivas:
- flips horizontais e verticais;
- rotações;
- alterações de intensidade;
- zoom e transformações espaciais.

## Inferência
- Test-Time Augmentation (TTA)
- Threshold tuning por classe

## Objetivo técnico
- combater class imbalance;
- reduzir overfitting;
- melhorar recall das classes minoritárias.

### F1-score macro
**≈ 0.5196**

---

# v100 — v99 + CLAHE

## Objetivo
Combinar:
- toda a pipeline avançada da v99;
- preprocessamento CLAHE aplicado ao dataset original.

## Melhorias herdadas da v99
- AdamW
- OneCycleLR
- Mixup
- Focal Loss
- Label smoothing
- TTA
- Threshold tuning
- Augmentações agressivas

## Nova melhoria
- CLAHE integrado diretamente no preprocessamento

## Objetivo clínico
Melhorar:
- contraste local;
- deteção de detalhes finos;
- robustez em estruturas anatómicas pouco visíveis.

## Configuração principal
- Input size: 256×256
- Batch size: 16
- Epochs: 80
- Learning rate: 3e-4

## Observações
A v100 representa a versão mais avançada da pipeline DenseNet disponível nos notebooks fornecidos.

### F1-score macro
Não identificado diretamente nos outputs do notebook disponibilizado.

---

# Evolução Global do Projeto

## Principais tendências observadas

### 1. Otimização computacional
A redução da resolução e o aumento do batch size permitiram:
- treino mais rápido;
- maior estabilidade;
- melhor aproveitamento do hardware.

### 2. Melhorias de preprocessamento
O CLAHE mostrou impacto relevante na melhoria do contraste em imagens médicas.

### 3. Estratégias anti-overfitting
Foram adicionadas:
- Mixup;
- augmentações agressivas;
- label smoothing;
- regularização via AdamW.

### 4. Otimização para F1-score
As versões mais avançadas focaram-se explicitamente em:
- class imbalance;
- aumento do recall;
- melhoria do F1-score macro.

### 5. Inferência avançada
A utilização de:
- TTA;
- threshold tuning;
permitiu melhorar robustez e estabilidade da classificação final.

---

# Tabela Resumo

| Versão | Principais melhorias | Input Size | Batch Size | Estratégias principais | F1-score macro |
|---|---|---|---|---|---|
| v0 | Baseline inicial | 512×512 | 4 | Pipeline base DenseNet | 0.5466 |
| v1 | Redução resolução + batch maior | 224×224 | 16 | Melhor estabilidade e velocidade | 0.5910 |
| v2 | Introdução de CLAHE | 512×512 | 4 | Melhor contraste médico | 0.6385 |
| v3 | CLAHE + resolução reduzida | 224×224 | 16 | Otimização computacional | 0.5515 |
| v99 | F1Boost | 512×512 | 8 | AdamW, OneCycleLR, Mixup, TTA | 0.6467 |
| v100 | v99 + CLAHE | 256×256 | 16 | Pipeline avançada + CLAHE | 0.7076 |

---

# Conclusão

A evolução do projeto demonstrou uma abordagem iterativa orientada à melhoria do desempenho em classificação médica.

As principais contribuições ao longo das versões foram:
- preprocessamento especializado;
- regularização avançada;
- técnicas modernas de otimização;
- estratégias focadas em class imbalance.
