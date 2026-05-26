# Léxico de Aprendizagem Profunda (Deep Learning)


## 1. Redes Neuronais e CNNs

### CNN (Convolutional Neural Network)

```
Rede neuronal especializada em processamento de imagens.
```

### DenseNet

```
Arquitetura de CNN onde cada camada recebe informação de todas as camadas anteriores.

Vantagens:
- reutilização de características;
- menor perda de informação;
- melhor fluxo de gradientes.
```
### Convolution

```
Operação matemática central das CNNs utilizada para extrair características das imagens (bordas, texturas, etc.).
```

### Weights

```
Parâmetros internos aprendidos pela rede neuronal durante o treino.
```

### Forward Pass

```
Passagem dos dados pela rede para produzir previsões.
```

### Backpropagation

```
Algoritmo utilizado para atualizar os pesos da rede neuronal com base no erro calculado.
```

### Gradient

```
Valor matemático que indica como ajustar os pesos para minimizar o erro.
```


## 2. Treino do Modelo

### Hyperparameters

```
Parâmetros definidos manualmente antes do treino.

Exemplos:
- learning rate: controla o tamanho das atualizações dos pesos durante o treino.
- batch size: número de imagens processadas de cada vez;
- epochs.
```

### Loss Function

```
Função matemática utilizada para medir o erro do modelo.
```

### CrossEntropy Loss

```
Função de loss muito utilizada em classificação multi-classe.
Mede a diferença entre:
- previsão do modelo;
- classe correta.
```

### Focal Loss

```
Função de loss criada para problemas com desequilíbrio de classes.
Dá mais importância às amostras difíceis.
```

### Optimizer

```
Algoritmo responsável por atualizar os pesos da rede neuronal.

Exemplos:
- SGD;
- Adam;
- AdamW.
```

### SGD (Stochastic Gradient Descent)

```
Otimizador clássico baseado em gradiente.
```

### Adam

```
Otimizador baseado em gradiente que combina:
- Momentum;
- RMSProp.

Muito utilizado em Deep Learning devido à sua estabilidade e rapidez de convergência.
```

### AdamW

```
Versão melhorada do Adam que separa o weight decay da atualização dos pesos, ajudando a reduzir overfitting.
```

### Scheduler

```
Mecanismo que ajusta automaticamente o learning rate durante o treino.
```

### OneCycleLR

```
Scheduler que altera dinamicamente o learning rate ao longo do treino.
Pode melhorar convergência e generalização.
```

### Callback

```
Função executada automaticamente durante o treino.

Exemplos:
- guardar melhor modelo;
- early stopping;
- logging.
```

### Checkpoint

```
Ficheiro guardado contendo:
- pesos do modelo;
- estado do treino;
- métricas.

Permite continuar ou recuperar treinos.
```

### Early Stopping

```
Técnica que interrompe o treino quando o modelo deixa de melhorar.
```


## 3. Métricas de Avaliação

### Metrics

```
Indicadores utilizados para avaliar o desempenho do modelo.

Exemplos:
- accuracy;
- precision;
- recall;
- F1-score.
```

### Accuracy

```
Percentagem de previsões corretas realizadas pelo modelo relativamente ao total de amostras.
```

### Precision

```
Métrica que mede quantas previsões positivas estavam corretas.

Fórmula: Precision = TP / (TP + FP)
```

### Recall

```
Métrica que mede quantos casos positivos reais foram corretamente identificados.

Fórmula: Recall = TP / (TP + FN)
```

### F1-score

```
Métrica que combina:
- precision;
- recall.

Muito útil em datasets desbalanceados.

Fórmula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### F1-score Macro

```
Média do F1-score calculado individualmente para todas as classes.
Todas as classes têm o mesmo peso.
```

### Macro Average

```
Média simples calculada igualmente para todas as classes.
```

### AUC (Area Under the Curve)

```
Métrica que mede a capacidade de separação entre classes.
Quanto mais próximo de 1, melhor o desempenho.
```

### Confusion Matrix

```
Tabela que compara:
- valores reais;
- previsões do modelo.

Permite analisar erros de classificação.
```


## 4. Preprocessamento e Data Augmentation

### Preprocessing

```
Transformações realizadas antes do treino.

Exemplos:
- resize;
- normalização;
- CLAHE.
```

### Resize

```
Alteração da resolução da imagem.
```

### Normalization

```
Processo de normalização dos valores da imagem.
Ajuda a estabilizar o treino.
```

### CLAHE

```
Contrast Limited Adaptive Histogram Equalization.
Técnica de melhoria de contraste em imagens médicas.

Melhora:
- contraste local;
- visibilidade de detalhes anatómicos.
```

### Augmentation (Data Augmentation)

```
Técnicas de transformação artificial das imagens para aumentar a variabilidade do dataset.

Exemplos:
- rotação;
- flip;
- zoom;
- alteração de brilho;
- crop.

Objetivo:
- reduzir overfitting;
- melhorar generalização.
```

### Mixup

```
Técnica de augmentation que mistura imagens e labels.
Ajuda na generalização.
```

### TTA (Test-Time Augmentation)

```
Aplicação de augmentações durante a inferência.

Objetivo:
- aumentar robustez;
- melhorar estabilidade das previsões.
```


## 5. Regularização e Generalização

### Overfitting

```
Situação em que o modelo aprende demasiado bem os dados de treino, mas falha em generalizar.
```

### Regularization

```
Conjunto de técnicas utilizadas para reduzir overfitting.

Exemplos:
- dropout;
- weight decay;
- augmentation.
```

### Dropout

```
Técnica de regularização que desativa neurónios aleatoriamente durante o treino.
Ajuda a reduzir overfitting.
```

### Weight Decay

```
Técnica de regularização que penaliza pesos muito elevados.
```

### Label Smoothing

```
Técnica que suaviza os labels para evitar excesso de confiança do modelo.
```

### Threshold Tuning

```
Ajuste do limiar de decisão utilizado para classificação.
Pode melhorar F1-score e recall.
```

### Desequilíbrio das Classes

```
Situação em que algumas classes possuem muito mais exemplos do que outras.
Pode prejudicar o treino do modelo.
```

### Class Weights

```
Pesos atribuídos às classes para compensar desequilíbrio das classes.
Classes raras recebem maior importância na loss.
```


## 6. Transfer Learning e Fine-Tuning

### Transfer Learning

```
Utilização de modelos pré-treinados em novos problemas.
```

### Fine-Tuning

```
Ajuste de um modelo pré-treinado para um novo problema específico.
```
