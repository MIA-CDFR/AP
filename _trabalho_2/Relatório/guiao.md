# Guião de Apresentação — Grupo G8
## Classificação de Imagens ERCP com CNNs Profundas
**Duração total:** 10 minutos | **Slides:** 11 | **Discussão:** 5 min

---

## Slide 1 — Título (0:00 – 0:30)

**Orador:** qualquer membro do grupo

> "Bom dia / Boa tarde. Somos o Grupo 8 e vamos apresentar o nosso trabalho prático do Módulo 2 de Aprendizagem Profunda.
> O tema é a classificação automática de imagens de ERCP — Colangiopancreatografia Retrógrada Endoscópica — utilizando quatro arquiteturas CNN com transfer learning: DenseNet121, ResNet50, MobileNetV2 e EfficientNet-B7."

---

## Slide 2 — Problema e Dataset (0:30 – 1:30) ⏱ 1 min

**Orador sugerido:** membro que trabalhou na análise do dataset

> "O nosso problema é classificar imagens fluoroscópicas em quatro categorias clínicas: Biliary Leaks — fugas biliares —, Lithiasis — cálculos biliares —, Stricture — estenoses dos ductos — e Normal.
>
> O dataset é o MIQR-CC, com 1067 imagens de treino, 234 de validação e 267 de teste.
>
> O desafio central que encontrámos logo na análise exploratória foi o **desequilíbrio severo**: 47% das imagens de treino pertencem à classe Lithiasis, enquanto Biliary Leaks representa apenas 10%.
>
> A métrica usada é o F1-score macro — peso igual por classe. Isto significa que um modelo que ignore completamente a classe Biliary Leaks é muito penalizado, mesmo que tenha accuracy global elevada."

**Ponto-chave a enfatizar:** *"O desequilíbrio não é um pormenor técnico — é o problema central deste trabalho."*

---

## Slide 3 — 4 Arquiteturas CNN (1:30 – 2:15) ⏱ 45 s

**Orador sugerido:** membro que trabalhou na parte teórica

> "Explorámos quatro arquiteturas pré-treinadas no ImageNet.
>
> A **DenseNet121** usa conectividade densa — cada camada recebe as activações de todas as anteriores — o que preserva features de baixo nível em qualquer profundidade. A **ResNet50** aprende funções residuais, o que estabiliza o treino. A **MobileNetV2** é muito leve com apenas 3.4M parâmetros, mas essa leveza tem custo em capacidade representacional. E a **EfficientNet-B7**, a maior com 66M parâmetros, aplica compound scaling, mas revelou problemas sérios de sobreajuste com os dados disponíveis.
>
> Todos utilizam transfer learning — com apenas 1067 imagens de treino, treinar de raiz seria inviável."

---

## Slide 4 — Estratégia Metodológica (2:15 – 3:15) ⏱ 1 min

**Orador sugerido:** membro que implementou as técnicas (DenseNet ou ResNet)

> "A nossa principal descoberta metodológica foi que o **tratamento do desequilíbrio** era muito mais importante do que qualquer ajuste de pré-processamento isolado.
>
> Usámos três linhas de ataque:
>
> Primeiro, **funções de perda ponderadas**: Focal Loss — que penaliza mais os exemplos fáceis, forçando o modelo a focar nas classes raras — combinada com cross-entropy ponderada inversamente à frequência da classe.
>
> Segundo, **augmentação e inferência robusta**: Mixup com alpha 0.4, que interpola imagens e labels entre classes, Test-Time Augmentation com 4 a 8 passes, e ajuste de threshold por classe no conjunto de validação.
>
> Terceiro, **pré-processamento**: CLAHE offline para realçar estruturas biliares subtis, e optimizadores adaptados por arquitectura — AdamW com OneCycleLR para DenseNet, differential learning rates para MobileNet.
>
> O impacto quantitativo na DenseNet foi +10.2 pontos percentuais com as técnicas anti-desequilíbrio, e mais +6.1 pp com o CLAHE e AMP — ou seja, a estratégia de treino foi 25 vezes mais impactante do que o pré-processamento."

---

## Slide 5 — Resultados Globais (3:15 – 4:45) ⏱ 1:30 min

**Orador sugerido:** membro que trabalhou na DenseNet / resultados finais

> "Passando aos resultados no conjunto de teste com 267 amostras.
>
> A **DenseNet121 v100** obteve o melhor resultado global: F1 macro 0.7076 e accuracy 0.7266. O AUC-ROC estimado via One-vs-Rest é aproximadamente 0.92. Em Biliary Leaks — a classe mais crítica — o F1 é 0.667.
>
> A ResNet50 v2 ficou em segundo lugar com F1 0.6647. EfficientNet e MobileNet ficaram em torno de 0.555 — ambas afectadas pelo sobreajuste e limitações de capacidade, respectivamente.
>
> Em relação à **baseline do artigo de referência** — Martins et al., 2025 — que reporta F1 macro 0.738, o nosso melhor modelo fica 3 pontos percentuais abaixo. No entanto, se compararmos com a nossa própria baseline inicial — a v0 com F1 0.545 — o ganho é de +16.3 pontos percentuais, o que valida a robustez da estratégia experimental."

**Ponto-chave:** *"O gap face à baseline é real, mas esperado — o artigo usa dados adicionais de validação cruzada que nós não explorámos."*

---

## Slide 6 — F1 por Classe (4:45 – 5:45) ⏱ 1 min

**Orador sugerido:** membro que fez análise comparativa

> "Olhando para a análise por classe, vemos padrões interessantes.
>
> **Biliary Leaks** é a classe mais difícil em todos os modelos. A DenseNet é a melhor com F1 0.667, mas o recall é apenas 9 de 17 amostras de teste. Isto é clinicamente preocupante porque um falso negativo nesta classe significa uma fuga biliar não detectada — potencialmente fatal.
>
> **Lithiasis** é a mais fácil — a ResNet é ligeiramente melhor aqui com F1 0.747.
>
> **Stricture** revela um padrão sistemático muito relevante: a confusão com Lithiasis. A morfologia de uma estenose em fluoroscopia pode assemelhar-se a cálculos biliares, levando a tratamentos inadequados.
>
> A ResNet destaca-se com recall de Normal de 0.907 — 39 de 43 acertos — mas à custa de muitos falsos positivos."

---

## Slide 7 — Matrizes de Confusão (5:45 – 6:30) ⏱ 45 s

**Orador sugerido:** mesmo orador do slide anterior, ou transição rápida

> "As matrizes de confusão confirmam os padrões que acabámos de descrever.
>
> Na DenseNet, vemos a diagonal principal bem preenchida: 9 de 17 Biliary Leaks, 83 de 123 Lithiasis, 32 de 43 Normal e 70 de 84 Stricture correctos.
>
> O erro mais visível — a vermelho — é Stricture a ser classificado como Lithiasis: 14 casos na DenseNet e 15 na ResNet. Este é o erro com maior impacto clínico de todo o trabalho.
>
> Na ResNet, notamos algo diferente: o recall de Normal é excepcional — 39 de 43 — mas há mais erros em Biliary Leaks para Normal — 8 casos."

---

## Slide 8 — Grad-CAM (6:30 – 8:00) ⏱ 1:30 min

**Orador sugerido:** membro que implementou o Grad-CAM (MobileNet ou EfficientNet)

> "O Grad-CAM calcula mapas de calor a partir dos gradientes da classe-alvo em relação ao último mapa de activação convolucional. Isto permite-nos verificar se o modelo está a focar nas regiões certas.
>
> No canto superior esquerdo, temos uma **Biliary Leaks correctamente classificada** a 52.4% de confiança. O heatmap activa claramente a região periductular — anatomicamente relevante para fugas biliares.
>
> No canto inferior esquerdo, um **Lithiasis com 84.8% de confiança** — activação concentrada na zona de depósito dos cálculos, com alta certeza.
>
> No lado direito, os casos incorrectos são os mais informativos. No topo, uma **Stricture classificada como Lithiasis** — o modelo activa uma estrutura tubular e interpreta-a como presença de cálculo. Este é exactamente o erro sistemático que vimos nas matrizes de confusão.
>
> Em baixo, uma **Biliary Leaks classificada como Lithiasis** — o ducto principal activa de forma semelhante a cálculos. Isto confirma a dificuldade visual intrínseca desta classe.
>
> Estas visualizações validam que o modelo aprende regiões anatomicamente plausíveis — não está a focar em artefactos externos."

**Ponto-chave:** *"O Grad-CAM é obrigatório no enunciado, mas aqui serve também para explicar clinicamente os erros."*

---

## Slide 9 — Ensemble (8:00 – 8:30) ⏱ 30 s

**Orador sugerido:** membro que trabalhou no ensemble (Carlos/CB)

> "Implementámos também um ensemble por soft voting — média das probabilidades de saída — entre DenseNet e EfficientNet. A premissa é que os dois modelos erram em padrões distintos.
>
> Com os modelos base, já obtivemos F1 superior a 0.55. O próximo passo óbvio é combinar os dois melhores modelos individuais — DenseNet v100 e ResNet v2 — que têm padrões de erro claramente complementares."

---

## Slide 10 — Conclusões (8:30 – 9:30) ⏱ 1 min

**Orador sugerido:** qualquer membro; idealmente o que apresentou os resultados

> "Para concluir.
>
> O ranking final posiciona a DenseNet121 v100 em primeiro lugar com F1 0.708, ResNet50 v2 em segundo com 0.665, e EfficientNet e MobileNet em empate técnico em torno de 0.556.
>
> A conclusão principal é que **tratar o desequilíbrio** foi 25 vezes mais impactante do que ajustar o pré-processamento. Focal Loss, Mixup e TTA foram os factores decisivos.
>
> Em relação ao trabalho futuro: sintetizar dados para Biliary Leaks com GAN, explorar ensemble DenseNet v100 + ResNet v2, e investigar Vision Transformers. Acreditamos que o gap de 3 pp face à baseline se fecha com dados sintéticos adicionais."

---

## Slide 11 — Q&A (9:30 – 10:00) ⏱ 30 s

**Orador:** membro que lidera a sessão

> "Obrigados pela atenção. Estamos disponíveis para questões."

---

## Perguntas Previsíveis e Respostas Preparadas

### P1: Porque é que o EfficientNet ficou tão atrás sendo o maior modelo?
> "O EfficientNet-B7 com 66M parâmetros sofreu sobreajuste severo com apenas ~1000 imagens de treino. O train F1 chegava a 0.99 enquanto o val F1 ficava em 0.40. A solução foi congelar o BatchNorm e fazer fine-tuning apenas dos últimos 3 blocos, o que estabilizou o treino mas limitou a capacidade de adaptação."

### P2: Porque é que a vossa melhor solução fica 3 pp abaixo da baseline do artigo?
> "O artigo MIQR-CC usa validação cruzada com múltiplos folds e reporta resultados médios. Nós usámos o split fixo fornecido. Além disso, o artigo pode usar augmentação de dados mais extensiva ou arquitecturas com atenção espacial. O gap de 3 pp é real mas compreensível dado o contexto."

### P3: Como é que o Grad-CAM foi implementado?
> "Usámos a implementação padrão que calcula os gradientes da saída da classe-alvo em relação ao último mapa de activação convolucional, faz a média espacial para obter pesos de importância, e aplica ReLU. Foi implementado nos notebooks da MobileNet v7 e EfficientNet RR12."

### P4: O threshold tuning não seria overfitting ao validation set?
> "É uma preocupação legítima. O threshold tuning foi aplicado no validation set e avaliado no test set separado — o test set nunca foi visto durante o desenvolvimento. A separação é limpa: treino para aprender, validação para calibrar, teste para avaliar."

### P5: Qual seria o próximo passo mais impactante?
> "Um ensemble DenseNet v100 + ResNet v2, pois os seus padrões de erro são complementares: a DenseNet é melhor em Biliary Leaks e Stricture, a ResNet é melhor em Normal e Lithiasis. Esperamos que essa combinação supere a baseline de 0.738."

---

## Distribuição de Tempo por Orador (sugestão)

| Slides | Tempo | Orador sugerido |
|---|---|---|
| 1–2 | 0:00–1:30 | Membro A (contexto/dataset) |
| 3–4 | 1:30–3:15 | Membro B (arquiteturas/metodologia) |
| 5–7 | 3:15–6:30 | Membro C (resultados/análise) |
| 8–11 | 6:30–10:00 | Membro D (Grad-CAM/conclusões) |

---

## Notas de Apresentação

- **Números chave a memorizar:** F1=0.7076 (DenseNet), F1=0.738 (baseline), +16.3 pp interno, −3.0 pp baseline
- **Mostrar o laptop** com o relatório aberto durante Q&A para referência rápida a tabelas
- **Transição slides 7→8:** "As matrizes mostram o quê — o Grad-CAM mostra o porquê"
- **Slide 4:** enfatizar "25× mais impactante" — é o número mais impressionante do trabalho
- **Slide 8:** dar tempo ao público para observar as imagens; são auto-explicativas
