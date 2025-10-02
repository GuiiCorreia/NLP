# Problema do XOR: Perceptron Simples vs. Rede de Duas Camadas

## 📋 Questão

**Mostre que um único perceptron não consegue aprender a função XOR. Depois, implemente uma rede de duas camadas que consiga resolvê-la.**

---

## 🎯 Objetivo

Demonstrar através de implementação prática:
1. **Por que um perceptron simples falha** ao aprender a função XOR
2. **Como uma rede de duas camadas (MLP)** consegue resolver o problema

---

## 📊 A Função XOR

A função XOR (ou exclusivo) retorna 1 apenas quando as entradas são diferentes:

| X₁ | X₂ | Y (XOR) |
|----|----|---------| 
| 0  | 0  | 0       |
| 0  | 1  | 1       |
| 1  | 0  | 1       |
| 1  | 1  | 0       |

---

## ❌ Parte 1: Por que o Perceptron Simples Falha?

### Problema Fundamental: Separabilidade Linear

Um perceptron simples implementa a função:

```
y = step(w₁x₁ + w₂x₂ + b)
```

Esta função representa uma **linha reta** (ou hiperplano em dimensões maiores) que divide o espaço em duas regiões.

### Limitação do Perceptron

Para classificar corretamente o XOR, seria necessário separar:
- **Classe 0**: pontos (0,0) e (1,1)
- **Classe 1**: pontos (0,1) e (1,0)

**Problema**: Esses pontos estão em configuração diagonal! **Não existe uma linha reta** que possa separar essas duas classes.

### Teorema de Convergência do Perceptron

O teorema de convergência do perceptron (Rosenblatt, 1962) garante convergência apenas para problemas **linearmente separáveis**. Como XOR não é linearmente separável, o perceptron nunca convergirá para uma solução correta.

### Resultado Esperado

O perceptron simples terá **acurácia ≤ 75%**, geralmente acertando apenas 2 ou 3 dos 4 casos.

---

## ✅ Parte 2: Solução com Rede de Duas Camadas (MLP)

### Arquitetura da Rede

```
Input Layer (2 neurônios)
        ↓
Hidden Layer (2 neurônios, sigmoid)
        ↓
Output Layer (1 neurônio, sigmoid)
```

### Como Funciona?

1. **Camada Oculta**: Transforma o espaço de entrada de forma não-linear
2. **Representação Intermediária**: Cria um novo espaço onde o problema se torna linearmente separável
3. **Camada de Saída**: Realiza a classificação final no espaço transformado

### Algoritmo de Treinamento

**Forward Pass:**
```
h = sigmoid(W₁X + b₁)  # Camada oculta
y = sigmoid(W₂h + b₂)   # Camada de saída
```

**Backward Pass (Backpropagation):**
```
δ₂ = (y - target) × sigmoid'(y)       # Erro da saída
δ₁ = (δ₂ × W₂) × sigmoid'(h)          # Erro da camada oculta

# Atualizar pesos
W₂ = W₂ - lr × δ₂ × h
W₁ = W₁ - lr × δ₁ × X
```

### Resultado Esperado

O MLP alcança **acurácia de 100%**, classificando corretamente todos os 4 casos de XOR.

---

## 🚀 Como Executar

### Requisitos

```bash
pip install numpy matplotlib
```

### Execução

```bash
python xor_neural_networks.py
```

### Saídas Geradas

O programa gera:

1. **Resultados no Terminal**:
   - Tabela verdade do XOR
   - Desempenho do perceptron simples (falha)
   - Desempenho do MLP (sucesso)

2. **Gráficos PNG**:
   - `perceptron_simples.png` - Fronteira de decisão do perceptron (inadequada)
   - `mlp_xor.png` - Fronteira de decisão do MLP (correta)
   - `convergencia_arquitetura.png` - Curva de convergência e arquitetura

---

## 📈 Resultados Esperados

### Perceptron Simples
- **Acurácia**: ~50-75%
- **Conclusão**: Não converge para solução correta
- **Razão**: XOR não é linearmente separável

### MLP (Duas Camadas)
- **Acurácia**: 100%
- **Épocas necessárias**: ~5000
- **Conclusão**: Resolve XOR perfeitamente
- **Razão**: Camada oculta cria representações não-lineares

---

## 🧠 Conceitos Importantes

### 1. Separabilidade Linear
Um problema é **linearmente separável** se existe um hiperplano que separa perfeitamente as classes.

**Exemplos**:
- ✅ AND, OR, NOT → Linearmente separáveis
- ❌ XOR, XNOR → Não linearmente separáveis

### 2. Representações Não-Lineares
A camada oculta do MLP aprende a transformar o espaço de entrada, criando características que tornam o problema separável na nova representação.

### 3. Teorema da Aproximação Universal
Uma rede neural com pelo menos uma camada oculta pode aproximar qualquer função contínua, dado neurônios suficientes (Cybenko, 1989).

### 4. Backpropagation
Algoritmo para calcular gradientes eficientemente em redes multicamadas, permitindo o treinamento através de otimização por gradiente descendente.

---

## 📚 Referências

1. **Rosenblatt, F. (1958)**. "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"

2. **Minsky, M. & Papert, S. (1969)**. "Perceptrons" - Demonstrou matematicamente as limitações do perceptron

3. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986)**. "Learning representations by back-propagating errors"

4. **Cybenko, G. (1989)**. "Approximation by superpositions of a sigmoidal function"

5. **Material do Curso**: [Neural Networks - Stanford](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)

---

## 💡 Conclusões

### Por que Perceptron Falha?
- Implementa apenas separação linear
- XOR requer separação não-linear
- Limitação fundamental da arquitetura

### Por que MLP Funciona?
- Camada oculta cria representações não-lineares
- Transforma o espaço tornando-o linearmente separável
- Backpropagation permite aprender os pesos adequados

### Implicações Históricas
O problema do XOR foi crucial na história das redes neurais:
- **1969**: Minsky & Papert mostraram limitações do perceptron → "AI Winter"
- **1986**: Rumelhart et al. introduziram backpropagation → Renascimento das redes neurais
- **Hoje**: Redes profundas (deep learning) resolvem problemas muito mais complexos

---

## 👨‍💻 Estrutura do Código

```
xor_neural_networks.py
├── Dados do XOR
├── Funções de Ativação (sigmoid, step)
├── Classe SimplePerceptron
│   ├── predict()
│   ├── train()
│   └── evaluate()
├── Classe TwoLayerNetwork (MLP)
│   ├── forward()
│   ├── backward()
│   ├── train()
│   └── predict()
├── Treinamento e Avaliação
└── Visualizações
```

---

## 🎓 Aprendizados Principais

1. **Nem todos os problemas são linearmente separáveis**
2. **Camadas ocultas adicionam poder computacional**
3. **Não-linearidade é essencial para problemas complexos**
4. **Backpropagation permite treinar redes profundas**
5. **Até problemas simples podem requerer arquiteturas não-triviais**

---

## 📝 Notas Adicionais

- O número mínimo de neurônios na camada oculta para resolver XOR é **2**
- Taxa de aprendizado típica: **0.5 para MLP, 0.1 para perceptron**
- Função sigmoid é usada para introduzir não-linearidade
- Para problemas mais complexos, podem ser necessárias mais camadas e neurônios

