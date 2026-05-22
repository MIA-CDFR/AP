# Relatório Técnico: Evolução Metodológica do Pipeline de Classificação Multi-Classe em Imagens de ERCP com Arquitectura ResNet50

**Disciplina:** Módulo 2 — Aprendizagem Profunda para Imagens Médicas  
**Data:** Maio de 2026  
**Artefactos analisados:** `RESNET.ipynb`, `RESNET_final_v1.ipynb`, `RESNET_final_v2.ipynb`, `RESNET_final_v3.ipynb`

---

## Resumo

O presente relatório documenta a evolução metodológica de um sistema de classificação de imagens médicas endoscópicas obtidas através de Colangiopancreatografia Retrógrada Endoscópica (ERCP). A arquitectura de base utilizada é uma Rede Neural Convolucional Residual de 50 camadas (`ResNet50`) pré-treinada na base de dados ImageNet, adaptada para a tarefa de classificação multi-classe. Partindo de um notebook experimental de referência (`RESNET.ipynb`), foram desenvolvidas três variantes finais que exploram distintas hipóteses de pré-processamento (genérico vs. guiado pelo domínio), realçamento de contraste offline (CLAHE) e regularização do classificador. Este relatório detalha os parâmetros modificados entre versões, a fundamentação técnica de cada decisão de design e a interpretação metodológica do papel de cada variante no processo experimental global.

**Palavras-chave:** aprendizagem profunda, classificação de imagens médicas, ERCP, ResNet50, transfer learning, data augmentation, regularização.

---

## 1. Introdução

### 1.1 Enquadramento

A classificação automática de imagens endoscópicas constitui uma tarefa clinicamente relevante no contexto do diagnóstico assistido por computador. Imagens de ERCP apresentam características visuais específicas — campo circular de visão endoscópica, bordas escuras e variação de iluminação intra-procedimento — que impõem desafios particulares a abordagens de aprendizagem profunda baseadas em modelos pré-treinados com imagens naturais.

A utilização de *Transfer Learning* a partir de modelos pré-treinados em conjuntos de dados de grande escala, como o ImageNet, é uma prática estabelecida na literatura quando o volume de dados do domínio-alvo é limitado (Tajbakhsh et al., 2016; Raghu et al., 2019). O sucesso desta abordagem depende criticamente das decisões de pré-processamento, estratégias de regularização e da configuração do protocolo de treino.

### 1.2 Objectivos do relatório

O presente documento tem como objectivos:

1. caracterizar o pipeline de referência (`RESNET.ipynb`) e identificar as suas limitações;
2. descrever sistematicamente as modificações introduzidas em cada versão final;
3. documentar todos os parâmetros alterados com os respectivos valores antes e depois de cada modificação;
4. fundamentar tecnicamente cada decisão de design à luz das boas práticas em aprendizagem profunda para imagens médicas;
5. posicionar cada variante experimental no contexto do processo de investigação global;
6. propor melhorias adicionais ao pipeline experimental.

### 1.3 Estrutura do relatório

O relatório organiza-se da seguinte forma: a Secção 2 apresenta a arquitectura e configuração de referência; a Secção 3 descreve a visão geral comparativa entre versões; as Secções 4, 5 e 6 analisam em detalhe cada variante final; a Secção 7 apresenta tabelas comparativas exaustivas; a Secção 8 apresenta os resultados experimentais obtidos; a Secção 9 propõe melhorias ao pipeline; a Secção 10 conclui com síntese metodológica; a Secção 11 lista as referências bibliográficas.

---

## 2. Pipeline de Referência: `RESNET.ipynb`

### 2.1 Arquitectura do modelo

O modelo de referência é uma `ResNet50` pré-treinada no ImageNet, disponibilizada através da biblioteca `torchvision`. A camada de classificação final (`fc`) foi substituída por uma camada linear adaptada ao número de classes do problema:

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_class)
```

Não foi aplicada qualquer forma de regularização explícita na camada de classificação, o que representa uma limitação reconhecida em cenários de dados médicos com dimensão reduzida.

### 2.2 Protocolo de treino

**Tabela 1 — Hiperparâmetros de treino do notebook de referência**

| Parâmetro | Valor | Descrição |
|---|---|---|
| `EPOCHS` | 60 | Número máximo de épocas |
| `LEARNING_RATE` | `1e-4` | Taxa de aprendizagem inicial |
| `batch_size` | 4 | Tamanho do lote de treino |
| Função de perda | `FocalLoss(to_onehot_y=True)` | Focal Loss com conversão one-hot |
| Optimizador | `Adam` | Optimizador adaptativo |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` | Decréscimo cossinal da taxa de aprendizagem |
| Paciência (early stopping) | 10 épocas | Critério de paragem antecipada |
| Métrica de monitorização | F1 macro (validação) | Critério de selecção do melhor modelo |
| Métricas complementares | Accuracy, ROC AUC | Métricas de avaliação secundárias |

A **Focal Loss** (Lin et al., 2017) é uma escolha justificada pelo desequilíbrio esperado entre classes em datasets médicos. Esta função de perda atribui pesos dinâmicos maiores a exemplos difíceis de classificar, mitigando o efeito de dominância das classes maioritárias. O scheduler **CosineAnnealingLR** promove um decréscimo suave da taxa de aprendizagem, favorecendo a convergência para mínimos de melhor qualidade comparativamente a estratégias de decréscimo por patamares fixos.

### 2.3 Pipeline de pré-processamento e *data augmentation*

**Pipeline de treino (referência):**
```python
Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((512, 512)),
    RandRotate(range_x=15, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    RandAdjustContrast(prob=0.5),
    RandGaussianNoise(prob=0.3, mean=0.0, std=0.01),
    NormalizeIntensity(),
    Lambda(repeat_if_needed),
    ToTensor()
])
```

A estratégia de *augmentation* adoptada é deliberadamente agressiva, com probabilidades elevadas de transformação (0.5 para rotação e zoom). Esta abordagem visa maximizar a diversidade aparente do conjunto de treino, porém pode introduzir artefactos geométricos que distorcem características anatomicamente relevantes.

### 2.4 Limitações identificadas no notebook de referência

1. **Portabilidade reduzida:** o caminho do dataset está codificado como caminho absoluto de sistema (`/mounts/monica/ERCP_image_classification/dataset`) e o dispositivo está fixado em `cuda:0`, inviabilizando a execução em ambientes distintos.
2. **Ausência de regularização no classificador:** a camada `fc` é linear simples, sem `Dropout`, aumentando o risco de sobreajuste em datasets de dimensão limitada.
3. **Pré-processamento genérico:** o pipeline não explora as características específicas das imagens endoscópicas (região circular de interesse, fundo escuro sem informação clínica).
4. **Gestão de artefactos deficiente:** o nome do ficheiro guardado (`resnet50.pth`) é genérico e pode ser sobrescrito inadvertidamente entre execuções.
5. **`auc_metric` como variável global:** a sua definição fora da função de treino introduz dependência de estado global que compromete a modularidade e a segurança do código.

Estas limitações constituem a motivação directa para o desenvolvimento das três variantes finais documentadas nas secções seguintes.

## 3. Visão Geral Comparativa das Variantes

**Tabela 2 — Posicionamento experimental de cada variante**

| Variante | Hipótese experimental | Pré-processamento | Classificador | Portabilidade |
|---|---|---|---|---|
| `RESNET.ipynb` | Baseline de referência | Genérico, augmentation agressiva | `Linear` simples | Baixa |
| `RESNET_final_v1.ipynb` | Pré-processamento guiado pelo domínio melhora a classificação | ROI anatómica, máscara circular, escalonamento por percentis | `Dropout(0.3) + Linear` | Alta |
| `RESNET_final_v2.ipynb` | Baseline com regularização adicional, sem especialização do domínio | Genérico, augmentation agressiva | `Dropout(0.3) + Linear` | Alta |
| `RESNET_final_v3.ipynb` | Impacto do CLAHE offline: pré-processamento do domínio com imagens pré-realçadas | ROI anatómica, máscara circular, escalonamento por percentis, imagens CLAHE | `Dropout(0.3) + Linear` | Alta |

A invariância intencional dos hiperparâmetros de treino entre todas as variantes (cf. Tabela 7) garante que diferenças nos resultados são atribuíveis exclusivamente às modificações de pré-processamento e arquitectura documentadas nas secções seguintes.

---

## 4. `RESNET_final_v1.ipynb` — Pré-processamento Guiado pelo Domínio

### 4.1 Motivação

A variante `v1` parte da hipótese de que o pré-processamento das imagens endoscópicas pode beneficiar de conhecimento explícito do domínio. Em imagens de ERCP, o campo de visão endoscópico possui geometria circular e a periferia da imagem corresponde tipicamente a uma região escura sem informação diagnóstica. A não remoção desta região pode introduzir ruído no processo de extracção de características, especialmente nas camadas convolucionais iniciais da `ResNet50`, que actuam sobre padrões de baixo nível e são sensíveis a estruturas periféricas.

### 4.2 Modificações ao pipeline de pré-processamento

O pipeline de treino da `v1` introduz quatro novos passos antes da redimensão final:

```python
CropForeground(select_fn=lambda x: x > BLACK_THRESHOLD, margin=0),
CenterSpatialCrop(roi_size=(CENTER_ROI, CENTER_ROI)),
ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
Lambda(mask_circular_field),
```

Onde `CENTER_ROI = 480` e `BLACK_THRESHOLD = 0`.

**Justificação técnica de cada passo:**

- **`CropForeground`** — remove automaticamente as margens compostas exclusivamente por pixéis de intensidade zero (fundo escuro), eliminando artefactos do processo de captura sem perda de informação clínica.
- **`CenterSpatialCrop(roi_size=(480, 480))`** — extrai uma região central de 480×480 pixéis, focando o processamento na zona com maior densidade de informação endoscópica.
- **`ScaleIntensityRangePercentiles(lower=1, upper=99, ...)`** — realiza normalização de contraste robusta a *outliers* de intensidade, recorrendo aos percentis 1% e 99% como limites de escalonamento. Esta abordagem é superior à normalização min-max em imagens médicas onde artefactos de iluminação podem criar valores extremos não representativos.
- **`mask_circular_field`** — aplica uma máscara binária circular centrada na imagem, zerando pixéis fora do campo de visão endoscópico. Alinha o pré-processamento computacional com a geometria física do endoscópio.

```python
def mask_circular_field(img):
    _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device),
        torch.arange(w, device=img.device),
        indexing="ij"
    )
    cy, cx = h / 2.0, w / 2.0
    radius = 0.5 * min(h, w)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)
    return img * mask.unsqueeze(0).to(img.dtype)
```

### 4.3 Ajuste dos parâmetros de *data augmentation*

Após o pipeline de pré-processamento especializado, os parâmetros de *data augmentation* foram conservadoramente ajustados. Transformações agressivas podem destruir informação que o pré-processamento procurou preservar.

**Tabela 3 — Comparação dos parâmetros de augmentation: Referência vs. `v1`**

| Transformação | Referência | `v1` | Razão da alteração |
|---|---|---|---|
| `RandRotate` — ângulo máximo | 15 | 5 | Reduzir distorção geométrica após crop anatómico |
| `RandRotate` — probabilidade | 0.5 | 0.25 | Menor exposição a rotações após normalização espacial |
| `RandZoom` — intervalo | [0.9, 1.1] | [0.95, 1.05] | Preservar a proporção da ROI normalizada |
| `RandZoom` — probabilidade | 0.5 | 0.25 | Consistente com redução geral da agressividade |
| `RandAdjustContrast` — probabilidade | 0.5 | 0.2 | Contraste já normalizado por percentis |
| `RandAdjustContrast` — gamma | não definido | (0.9, 1.1) | Controlo explícito do intervalo de variação |
| `RandGaussianNoise` — probabilidade | 0.3 | 0.15 | Evitar ruído artificial após denoising implícito |
| `RandGaussianNoise` — std | 0.01 | 0.005 | Menor magnitude de perturbação |

### 4.4 Regularização do classificador

```python
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, num_class)
)
```

O *Dropout* (Srivastava et al., 2014) introduz ruído estocástico durante o treino, forçando a rede a aprender representações distribuídas e robustas. Em datasets médicos de dimensão limitada, esta técnica é considerada prática padrão para mitigar sobreajuste.

### 4.5 Adaptações de portabilidade

- `base_dir` alterado para relativo (`./dataset`).
- Dispositivo dinâmico: `torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")`.

### 4.6 Posicionamento metodológico

A variante `v1` testa a hipótese de que a incorporação de conhecimento do domínio médico no pré-processamento melhora a discriminabilidade das representações aprendidas. É a abordagem mais especializada e menos agnóstica ao domínio do conjunto de variantes.

---

## 5. `RESNET_final_v2.ipynb` — Baseline com Regularização

### 5.1 Motivação

A variante `v2` parte de um princípio complementar ao da `v1`: em vez de especializar o pipeline de pré-processamento, adicionar regularização ao classificador mantendo o pipeline genérico da referência. Esta abordagem permite isolar o efeito da regularização por *Dropout* — comparando directamente os resultados da `v2` com a referência (sem *Dropout*, pipeline genérico) é possível quantificar o contributo desta técnica; comparando `v2` com `v1` (ambos com *Dropout*, mas com pipelines de pré-processamento distintos) é possível isolar o efeito do pré-processamento especializado do domínio.

### 5.2 Pipeline de pré-processamento

Idêntico ao da referência (pipeline genérico sem especialização de domínio).

### 5.3 Regularização do classificador

```python
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, num_class)
)
```

A adição de `Dropout(p=0.3)` antes da camada de classificação linear introduz regularização estocástica durante o treino, forçando a rede a aprender representações distribuídas e robustas. Em datasets médicos de dimensão limitada, esta técnica é considerada prática padrão para mitigar sobreajuste (Srivastava et al., 2014).

### 5.4 Gestão de artefactos

```python
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)
model_name = os.path.join(MODEL_DIR, 'resnet50_v2.pth')
```

A utilização de `os.path.join` garante compatibilidade multi-plataforma. O nome `resnet50_v2.pth` identifica univocamente a variante e evita colisões com os artefactos das restantes versões.

### 5.5 Posicionamento metodológico

A variante `v2` serve como **controlo com regularização e pipeline genérico**: compara-se com a referência para isolar o efeito do *Dropout*; compara-se com `v1` para isolar o efeito do pré-processamento guiado pelo domínio (ambos com *Dropout*).

---

## 6. `RESNET_final_v3.ipynb` — Pré-processamento Guiado pelo Domínio com Imagens CLAHE

### 6.1 Motivação

A variante `v3` testa a hipótese de que aplicar *Contrast Limited Adaptive Histogram Equalization* (CLAHE) como etapa de pré-processamento offline melhora a qualidade das imagens de entrada, potenciando o desempenho do modelo em imagens endoscópicas com variação de iluminação. Mantém integralmente o pipeline especializado da `v1` — incluindo a extracção de ROI anatómica, a máscara circular e o escalonamento por percentis — e o mesmo classificador com `Dropout(0.3)`. A única diferença face à `v1` é a origem das imagens: `./dataset_clahe` contém as imagens pré-processadas com CLAHE, em vez de `./dataset` com as imagens originais.

### 6.2 Diferença face a `v1`

| Elemento | `v1` | `v3` |
|---|---|---|
| Directório de dados | `./dataset` | `./dataset_clahe` |
| Nome do modelo | `resnet50_v1.pth` | `resnet50_v3.pth` |
| Pipeline de transforms | Especializado (domínio) | Idêntico ao `v1` |
| Classificador | `Dropout(0.3) + Linear` | `Dropout(0.3) + Linear` |

Esta diferença minimal permite isolar com rigor o efeito do CLAHE: qualquer diferença de desempenho entre `v1` e `v3` é atribuível exclusivamente ao realçamento de contraste pré-aplicado nas imagens.

### 6.3 Pipeline de pré-processamento

Idêntico ao da `v1` (pré-processamento especializado do domínio, augmentation conservadora). Ver Secção 4.2 e Tabela 3 para descrição detalhada dos passos e justificação técnica.

### 6.4 Regularização do classificador

Idêntica à `v1` e `v2`:

```python
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, num_class)
)
```

### 6.5 Posicionamento metodológico

A variante `v3` funciona como **ablação do CLAHE offline**: mantém todo o pipeline da `v1` e altera exclusivamente o conjunto de imagens de entrada. Comparar `v3` com `v1` quantifica o efeito do realçamento de contraste adaptativo em pré-processamento, mantendo constantes todas as restantes variáveis; comparar `v3` com `v2` avalia o impacto combinado do pré-processamento especializado do domínio e do CLAHE face ao pipeline genérico.

---

## 7. Tabelas Comparativas Exaustivas

### 7.1 Configuração do ambiente e gestão de artefactos

**Tabela 4 — Comparação de configurações de ambiente**

| Elemento | Referência | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| Caminho do dataset | absoluto (remoto) | `./dataset` | `./dataset` | `./dataset_clahe` |
| Dispositivo de computação | `cuda:0` (fixo) | `mps/cpu` (dinâmico) | `mps/cpu` (dinâmico) | `mps/cpu` (dinâmico) |
| Criação automática do directório | não | `os.makedirs` | `os.makedirs` | `os.makedirs` |
| Nome do modelo guardado | `resnet50.pth` | `resnet50_v1.pth` | `resnet50_v2.pth` | `resnet50_v3.pth` |
| Modelo encapsulado em função | não | não | não | não |
| `auc_metric` dentro da função de treino | não | não | não | não |
| `map_location` no `torch.load` | não | não | não | não |

### 7.2 Pipeline de pré-processamento comparado

**Tabela 5 — Passos de pré-processamento no pipeline de treino**

| Passo | Referência | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| `LoadImage` + `EnsureChannelFirst` | ✓ | ✓ | ✓ | ✓ |
| `CropForeground` | — | ✓ `(x > 0)` | — | ✓ `(x > 0)` |
| `CenterSpatialCrop(480×480)` | — | ✓ | — | ✓ |
| `ScaleIntensityRangePercentiles(1%, 99%)` | — | ✓ | — | ✓ |
| Máscara circular (`mask_circular_field`) | — | ✓ | — | ✓ |
| Imagens com CLAHE pré-aplicado (offline) | — | — | — | ✓ |
| `Resize(512×512)` | ✓ | ✓ | ✓ | ✓ |
| `RandRotate` | `range=15, p=0.5` | `range=5, p=0.25` | `range=15, p=0.5` | `range=5, p=0.25` |
| `RandZoom` | `[0.9,1.1], p=0.5` | `[0.95,1.05], p=0.25` | `[0.9,1.1], p=0.5` | `[0.95,1.05], p=0.25` |
| `RandAdjustContrast` | `p=0.5` | `p=0.2, γ∈[0.9,1.1]` | `p=0.5` | `p=0.2, γ∈[0.9,1.1]` |
| `RandGaussianNoise` | `p=0.3, σ=0.01` | `p=0.15, σ=0.005` | `p=0.3, σ=0.01` | `p=0.15, σ=0.005` |
| `NormalizeIntensity` | ✓ | ✓ | ✓ | ✓ |
| `repeat_if_needed` + `ToTensor` | ✓ | ✓ | ✓ | ✓ |

### 7.3 Arquitectura do classificador

**Tabela 6 — Configuração da camada de classificação**

| Elemento | Referência | `v1` | `v2` | `v3` |
|---|---|---|---|---|
| Backbone | ResNet50 (ImageNet) | igual | igual | igual |
| Camada(s) de classificação | `Linear(2048, C)` | `Dropout(0.3) → Linear(2048, C)` | `Dropout(0.3) → Linear(2048, C)` | `Dropout(0.3) → Linear(2048, C)` |
| Encapsulada em `create_model()` | não | não | não | não |

*C = número de classes do problema ERCP*

### 7.4 Protocolo de treino (invariante entre versões)

**Tabela 7 — Hiperparâmetros de treino comuns a todas as versões**

| Parâmetro | Valor | Justificação da invariância |
|---|---|---|
| `EPOCHS` | 60 | Permite comparação directa entre versões |
| `LEARNING_RATE` | `1e-4` | Valor empiricamente estabelecido no baseline |
| `batch_size` | 4 | Limitado pela memória disponível |
| Função de perda | `FocalLoss(to_onehot_y=True)` | Robusta a desequilíbrio de classes |
| Optimizador | `Adam` | Optimizador adaptativo estável |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` | Decréscimo suave sem ajuste manual |
| Paciência (early stopping) | 10 épocas | Paragem conservadora |
| Métrica de early stopping | F1 macro (validação) | Equitativa para classes desequilibradas |
| Semente aleatória | 42 | Reprodutibilidade determinística |

A invariância intencional destes parâmetros garante que qualquer diferença observada nos resultados é atribuível exclusivamente às modificações de pré-processamento e arquitectura documentadas nas Tabelas 3–6.

---

## 8. Resultados Experimentais e Análise Comparativa

### 8.1 Métricas globais no conjunto de teste

O conjunto de teste é composto por **267 amostras** distribuídas pelas quatro classes: Biliary Leaks (17), Lithiasis (123), Normal (43), Stricture (84).

**Tabela 8 — Métricas globais de teste por variante**

| Variante | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| `RESNET.ipynb` (baseline) | 0.6367 | 0.6165 | 0.6370 |
| `RESNET_final_v1.ipynb` | 0.6854 | 0.6477 | 0.6854 |
| `RESNET_final_v2.ipynb` | 0.7041 | 0.6647 | 0.7157 |
| `RESNET_final_v3.ipynb` | 0.6404 | 0.6093 | 0.6202 |

A `v2` (pipeline genérico + Dropout) alcança o melhor desempenho global nos três indicadores. A `v1` supera o baseline em todas as métricas mas fica abaixo da `v2`. A `v3` (pipeline da `v1` com imagens CLAHE) regride face à `v1` em accuracy (−4.5 pp) e F1 macro (−3.8 pp), aproximando-se do baseline, o que sugere que o CLAHE offline não beneficia — e pode prejudicar — o pipeline especializado.

### 8.2 Métricas por classe no conjunto de teste

**Tabela 9 — F1-score por classe**

| Classe | Suporte | Baseline | `v1` | `v2` | `v3` |
|---|---|---|---|---|---|
| Biliary Leaks | 17 | 0.5625 | 0.6250 | 0.5517 | **0.7059** |
| Lithiasis | 123 | 0.6667 | 0.7352 | **0.7467** | 0.7224 |
| Normal | 43 | 0.6186 | 0.5227 | 0.5954 | 0.4878 |
| Stricture | 84 | 0.6182 | 0.7081 | **0.7651** | 0.5210 |

**Tabela 10 — Precision e Recall por classe**

| Variante | Classe | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | Biliary Leaks | 0.6000 | 0.5294 | 0.5625 |
| | Lithiasis | 0.6838 | 0.6504 | 0.6667 |
| | Normal | 0.5556 | 0.6977 | 0.6186 |
| | Stricture | 0.6296 | 0.6071 | 0.6182 |
| `v1` | Biliary Leaks | 0.6667 | 0.5882 | 0.6250 |
| | Lithiasis | 0.7154 | 0.7561 | 0.7352 |
| | Normal | 0.5111 | 0.5349 | 0.5227 |
| | Stricture | 0.7403 | 0.6786 | 0.7081 |
| `v2` | Biliary Leaks | 0.6667 | 0.4706 | 0.5517 |
| | Lithiasis | 0.8235 | 0.6829 | 0.7467 |
| | Normal | 0.4432 | 0.9070 | 0.5954 |
| | Stricture | 0.8769 | 0.6786 | 0.7651 |
| `v3` | Biliary Leaks | 0.7059 | 0.7059 | 0.7059 |
| | Lithiasis | 0.6136 | 0.8780 | 0.7224 |
| | Normal | 0.5128 | 0.4651 | 0.4878 |
| | Stricture | 0.8857 | 0.3690 | 0.5210 |

### 8.3 Matrizes de confusão

Ordem das classes: Biliary Leaks (BL), Lithiasis (Li), Normal (No), Stricture (St).

**Baseline (`RESNET.ipynb`)**

|  | Pred BL | Pred Li | Pred No | Pred St |
|---|---|---|---|---|
| **Real BL** | 9 | 0 | 7 | 1 |
| **Real Li** | 3 | 80 | 14 | 26 |
| **Real No** | 0 | 10 | 30 | 3 |
| **Real St** | 3 | 27 | 3 | 51 |

**`RESNET_final_v1.ipynb`**

|  | Pred BL | Pred Li | Pred No | Pred St |
|---|---|---|---|---|
| **Real BL** | 10 | 2 | 5 | 0 |
| **Real Li** | 1 | 93 | 14 | 15 |
| **Real No** | 1 | 14 | 23 | 5 |
| **Real St** | 3 | 21 | 3 | 57 |

**`RESNET_final_v2.ipynb`**

|  | Pred BL | Pred Li | Pred No | Pred St |
|---|---|---|---|---|
| **Real BL** | 8 | 1 | 8 | 0 |
| **Real Li** | 4 | 84 | 29 | 6 |
| **Real No** | 0 | 2 | 39 | 2 |
| **Real St** | 0 | 15 | 12 | 57 |

**`RESNET_final_v3.ipynb`**

|  | Pred BL | Pred Li | Pred No | Pred St |
|---|---|---|---|---|
| **Real BL** | 12 | 1 | 3 | 1 |
| **Real Li** | 1 | 108 | 13 | 1 |
| **Real No** | 2 | 19 | 20 | 2 |
| **Real St** | 2 | 48 | 3 | 31 |

### 8.4 Análise comparativa

Observações principais com base no conjunto completo de resultados:

1. **Dropout é o factor mais impactante (melhor desempenho global):** a comparação baseline vs. `v2` (único factor diferenciador: Dropout) mostra um ganho de +3.7 pp em accuracy e +4.8 pp em F1 macro. A `v2` é a variante com melhor accuracy (0.7041), melhor F1 macro (0.6647) e melhor F1 weighted (0.7157).

2. **Pré-processamento especializado melhora classes estruturais:** a `v1` apresenta os melhores F1 individuais em Stricture (0.7081, +9.0 pp vs. baseline) e forte melhoria em Lithiasis (0.7352, +6.9 pp). O isolamento da ROI anatómica beneficia classes com padrões morfológicos mais distintos e circunscritos.

3. **Classe Normal é problemática para o pré-processamento especializado:** tanto a `v1` (0.5227) como a `v3` (0.4878) degradam o F1 da classe Normal face ao baseline (0.6186). A máscara circular e o crop podem remover contexto periférico que distingue imagens normais das patológicas.

4. **`v2` sobre-prevê a classe Normal:** recall de 0.9070 com precision de 0.4432. O modelo sem especialização de domínio classifica como Normal uma fracção elevada de exemplos, penalizando as restantes classes.

5. **CLAHE offline não melhora o pipeline especializado (v3 vs. v1):** a `v3` regride face à `v1` em accuracy (0.6404 vs. 0.6854, −4.5 pp), F1 macro (0.6093 vs. 0.6477, −3.8 pp) e F1 de Stricture (0.5210 vs. 0.7081, −18.7 pp). O colapso no recall de Stricture (0.3690 — apenas 31 de 84 amostras correctas) e o enviesamento para Lithiasis (108/123 correctas, mas 48 Strictures mal classificadas como Lithiasis) sugerem que o CLAHE amplifica características espectrais que tornam Stricture visualmente semelhante a Lithiasis.

6. **`v3` melhora Biliary Leaks:** única classe em que o CLAHE é claramente benéfico (F1 0.7059 vs. 0.6250 na `v1`, melhor de todas as variantes). O realçamento de contraste pode tornar as lesões de fuga mais salientes.

---

## 9. Propostas de Melhoria ao Pipeline Experimental

Com base na análise crítica das variantes existentes, identificam-se as seguintes melhorias prioritárias:

### 9.1 Tratamento explícito do desequilíbrio de classes

O pipeline actual não aborda explicitamente o desequilíbrio entre classes (e.g., a sub-representação da classe `Biliary_Leaks`). Recomenda-se:

- **Pesos por classe na Focal Loss:** calcular pesos inversamente proporcionais à frequência de cada classe:
  ```python
  class_counts = np.bincount(data['train']['labels'])
  class_weights = 1.0 / class_counts
  class_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(device)
  loss_function = FocalLoss(to_onehot_y=True, weight=class_weights)
  ```
- **`WeightedRandomSampler`:** amostrar proporcionalmente para garantir que classes minoritárias são representadas em cada época com frequência adequada.

### 9.2 Validação cruzada estratificada

A utilização de uma divisão estática `train/val/test` não fornece estimativas robustas da variância dos resultados. Sugere-se a adopção de **validação cruzada estratificada k-fold** (k=5), especialmente relevante em datasets de dimensão limitada, para obter intervalos de confiança sobre as métricas reportadas.

### 9.3 Análise sistemática de *learning curves*

Adicionar análise das curvas de aprendizagem (loss e F1 em treino e validação por época) para diagnosticar sobreajuste ou sub-ajuste e calibrar os hiperparâmetros de regularização. O uso de `livelossplot` já está implementado, mas não é explorado sistematicamente para diagnóstico.

### 9.4 Extensão do protocolo de treino para a variante `v1`

Dada a maior complexidade do pipeline de pré-processamento da `v1`, o modelo pode necessitar de mais épocas para convergir. Sugere-se aumentar `EPOCHS` para 100 e a paciência para 15–20 épocas para esta variante.

### 9.5 Métricas por classe nos relatórios experimentais

Reportar métricas por classe (`precision`, `recall`, `F1` por classe) além das métricas globais, de forma a identificar classes problemáticas e fundamentar iterações futuras. O `classification_report` da `sklearn` já produz estes valores; recomenda-se guardá-los sistematicamente.

### 9.6 Incorporação de técnicas de explicabilidade

Para fins académicos e clínicos, sugere-se a aplicação de técnicas de *explainability* como **Grad-CAM** (Selvaraju et al., 2017) para visualizar as regiões de activação do modelo. Esta análise é especialmente relevante para validar se o pré-processamento anatómico da `v1` efectivamente orienta a atenção do modelo para regiões clinicamente relevantes.

---

## 10. Síntese e Conclusões

### 10.1 Síntese das contribuições de cada variante

As três variantes finais exploram dimensões experimentais complementares:

- **`RESNET_final_v1.ipynb`** — testa o impacto da incorporação de conhecimento do domínio médico no pré-processamento. A hipótese central é que a remoção explícita de ruído contextual e a normalização adaptativa de contraste melhoram a qualidade das representações aprendidas pela rede.

- **`RESNET_final_v2.ipynb`** — adiciona regularização por *Dropout* ao classificador mantendo o pipeline genérico da referência. Permite isolar o efeito da regularização (comparação com a referência) e o efeito do pré-processamento especializado (comparação com `v1`).

- **`RESNET_final_v3.ipynb`** — replica o pipeline especializado da `v1` utilizando o conjunto de imagens pré-realçadas por CLAHE (`dataset_clahe`). A hipótese central é que o realçamento adaptativo de contraste aplicado offline melhora a qualidade da informação de entrada. Comparar `v3` com `v1` isola exclusivamente o efeito do CLAHE.

### 10.2 Decisões de design partilhadas

As três variantes convergem em: (a) arquitectura `ResNet50` com pré-treino ImageNet; (b) `FocalLoss` como função de perda; (c) protocolo `Adam` + `CosineAnnealingLR`; (d) critério de selecção por F1 macro. Esta invariância intencional garante a comparabilidade entre variantes.

### 10.3 Limitações persistentes

- Ausência de pesos por classe ou sobreamostragem para classes minoritárias.
- Divisão estática `train/val/test` sem validação cruzada.
- Análise de convergência não sistemática.
- Métricas por classe não reportadas de forma estruturada.

### 10.4 Enquadramento para relatório ou dissertação académica

Para fins de documentação em dissertação ou relatório de avaliação, sugere-se apresentar as variantes segundo o seguinte enquadramento:

> *"A partir de um notebook de referência (`RESNET.ipynb`), foram desenvolvidas três variantes experimentais. A variante `v1` incorpora pré-processamento guiado pelo domínio endoscópico e regularização por Dropout; a variante `v2` aplica apenas Dropout mantendo o pipeline genérico, permitindo isolar o efeito do pré-processamento especializado; a variante `v3` replica o pipeline da `v1` com imagens pré-realçadas por CLAHE offline, quantificando o impacto desta técnica de melhoria de contraste. Os hiperparâmetros de treino foram mantidos constantes entre todas as variantes, de forma a isolar o efeito das modificações de pré-processamento e arquitectura nos resultados finais."*

---

## 11. Referências

- **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. *Proceedings of CVPR 2016*, pp. 770–778.
- **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).** Focal loss for dense object detection. *Proceedings of ICCV 2017*, pp. 2980–2988.
- **Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S. (2019).** Transfusion: Understanding transfer learning for medical imaging. *NeurIPS 2019*, pp. 3347–3357.
- **Selvaraju, R. R., et al. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV 2017*, pp. 618–626.
- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(1), 1929–1958.
- **Tajbakhsh, N., et al. (2016).** Convolutional neural networks for medical image analysis: Full training or fine tuning? *IEEE Transactions on Medical Imaging, 35*(5), 1299–1312.
- **MONAI Consortium (2020–2024).** MONAI: Medical Open Network for AI. Disponível em: https://monai.io