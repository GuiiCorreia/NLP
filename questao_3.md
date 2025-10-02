# Comparação Empírica de Funções de Ativação em Redes Neurais

## 📋 Questão 3

**Objetivo:** Compare empiricamente o desempenho de redes pequenas usando sigmoid, tanh e ReLU em uma mesma tarefa simples (ex.: classificação binária). Qual ativação levou a melhor convergência?

---

## 🎯 Metodologia

### Tarefa Escolhida
**Problema XOR** - Classificação binária não-linear

O problema XOR é ideal para este experimento porque:
- É simples mas não-linearmente separável
- Requer que a rede aprenda representações não-triviais
- É um benchmark clássico para redes neurais
- Possui solução conhecida e determinística

**Dados de entrada:**
```
Input (x1, x2) → Output (y)
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0
```

### Arquitetura da Rede
```
Camada de Entrada:  2 neurônios (x1, x2)
Camada Oculta:      4 neurônios (com ativação testada)
Camada de Saída:    1 neurônio (sigmoid para probabilidade)
```

**Parâmetros de treinamento:**
- Learning rate: 0.5
- Loss function: MSE (Mean Squared Error)
- Otimizador: Gradient Descent (batch)
- Épocas máximas: 1000
- Número de execuções: 5 (para robustez estatística)

### Funções de Ativação Testadas

#### 1. **Sigmoid**
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x) * (1 - σ(x))
```
- **Range:** (0, 1)
- **Características:** Suave, diferenciável, mas sofre de vanishing gradient

#### 2. **Tanh (Tangente Hiperbólica)**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```
- **Range:** (-1, 1)
- **Características:** Centralizada em zero, gradientes mais fortes que sigmoid

#### 3. **ReLU (Rectified Linear Unit)**
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 se x > 0, 0 caso contrário
```
- **Range:** [0, +∞)
- **Características:** Não satura, computacionalmente eficiente, pode sofrer de dying ReLU

---

## 🔧 Como Executar

### Requisitos
```bash
pip install numpy matplotlib seaborn
```

### Execução
```bash
python activation_comparison.py
```

### Saídas Geradas
1. **activation_comparison_results.png** - Gráficos comparativos
2. **relatorio_experimento.txt** - Estatísticas detalhadas

---

## 📊 Resultados Esperados

### Métricas Avaliadas

1. **Velocidade de Convergência**
   - Número de épocas até atingir 100% de acurácia
   - Métrica principal para determinar qual ativação é "melhor"

2. **Perda (Loss) ao Longo do Tempo**
   - Quão rapidamente o erro diminui
   - Indicador de estabilidade do treinamento

3. **Acurácia ao Longo do Tempo**
   - Progressão do aprendizado
   - Suavidade da curva de aprendizado

4. **Magnitude dos Gradientes**
   - Detecta vanishing/exploding gradients
   - Importante para entender estabilidade

### Interpretação dos Gráficos

#### Gráfico 1: Perda ao Longo do Treinamento
- **Eixo Y (log):** Loss (MSE)
- **Observar:** Qual curva decresce mais rapidamente
- **Interpretação:** Descida mais rápida = convergência mais rápida

#### Gráfico 2: Acurácia ao Longo do Treinamento
- **Eixo Y:** Acurácia (%)
- **Observar:** Qual atinge 100% primeiro
- **Interpretação:** Métrica direta de desempenho

#### Gráfico 3: Magnitude dos Gradientes
- **Eixo Y (log):** Norma do gradiente
- **Observar:** Estabilidade e magnitude
- **Interpretação:** 
  - Muito baixo = vanishing gradient
  - Muito alto = exploding gradient
  - Estável = bom fluxo de informação

#### Gráfico 4: Box Plot de Convergência
- **Observar:** Mediana, quartis, outliers
- **Interpretação:** Consistência entre diferentes execuções

---

## 🎓 Base Teórica

### Por que ReLU geralmente converge mais rápido?

1. **Não saturação:** 
   - ReLU não satura para valores positivos
   - Derivada constante = 1 para x > 0
   - Não sofre de vanishing gradient

2. **Sparsidade:**
   - Neurônios com entrada negativa não ativam
   - Rede mais eficiente e especializada

3. **Computacionalmente eficiente:**
   - Operação simples (max)
   - Derivada trivial de calcular

### Por que Tanh é melhor que Sigmoid?

1. **Centralização em zero:**
   - Saídas entre -1 e 1
   - Facilita aprendizado nas camadas seguintes
   - Gradientes melhor balanceados

2. **Gradientes mais fortes:**
   - Derivada máxima = 1 (em x=0)
   - Sigmoid tem derivada máxima = 0.25
   - Fluxo de gradiente 4x melhor

### Por que Sigmoid tem dificuldades?

1. **Vanishing Gradient:**
   - Derivada muito pequena nas extremidades
   - Dificulta aprendizado em redes profundas
   - Saturação em valores extremos

2. **Não centralizada:**
   - Saídas sempre positivas [0,1]
   - Pode causar zigzagging no gradiente descent
   - Convergência mais lenta

---

## 📈 Análise dos Resultados

### Ranking Esperado (do melhor ao pior)

**1º lugar: ReLU** ⚡
- Convergência mais rápida (~50-150 épocas)
- Gradientes estáveis
- Sem saturação

**2º lugar: Tanh** 🌊
- Convergência moderada (~100-300 épocas)
- Melhor que sigmoid
- Centralização ajuda

**3º lugar: Sigmoid** 🐌
- Convergência mais lenta (~300-600 épocas)
- Vanishing gradient
- Menos eficiente

### Fatores que Podem Afetar os Resultados

1. **Inicialização dos pesos:**
   - He initialization para ReLU
   - Xavier initialization para Sigmoid/Tanh

2. **Learning rate:**
   - Valor fixo (0.5) pode não ser ótimo para todas

3. **Problema específico:**
   - XOR é relativamente simples
   - Resultados podem variar em problemas mais complexos

---

## 🔬 Experimentos Adicionais (Opcional)

### Para Aprofundar

1. **Variar Learning Rate:**
   ```python
   for lr in [0.1, 0.5, 1.0]:
       results = run_experiment(learning_rate=lr)
   ```

2. **Arquiteturas Diferentes:**
   ```python
   # Testar: 2→8→1, 2→16→1, 2→4→4→1
   ```

3. **Outros Problemas:**
   - Moons dataset
   - Circles dataset
   - Classificação multi-classe

---

## 📚 Referências

1. **Material Stanford:** [Neural Networks Slides](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)

2. **Papers Clássicos:**
   - Glorot & Bengio (2010) - "Understanding the difficulty of training deep feedforward neural networks"
   - He et al. (2015) - "Delving Deep into Rectifiers: Surpassing Human-Level Performance"
   - LeCun et al. (2012) - "Efficient BackProp"

3. **Livros:**
   - Goodfellow, Bengio & Courville - "Deep Learning" (Cap. 6)
   - Nielsen - "Neural Networks and Deep Learning" (Cap. 3)
