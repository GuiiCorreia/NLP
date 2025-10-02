# MLP de 2 Camadas: Forward Pass e Backpropagation Manual

Este projeto demonstra o cálculo manual de **forward pass** (propagação para frente) e **backpropagation** (propagação para trás) em uma rede neural Multi-Layer Perceptron (MLP) de 2 camadas.

## 📚 Baseado no Material

Este código é baseado nos slides de Stanford sobre Redes Neurais:
- **Material**: [Neural Networks - Stanford](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)
- **Conceitos cobertos**: Forward pass, Backward differentiation, Computation graphs, Chain rule

## 🏗️ Arquitetura da Rede

```
Entrada (3 neurônios)
    ↓ W[1], b[1]
Camada Oculta (2 neurônios) - Ativação: ReLU
    ↓ W[2], b[2]
Saída (1 neurônio) - Ativação: Sigmoid
    ↓
Loss: Binary Cross-Entropy
```

### Dimensões dos Pesos

- **W[1]**: [2 × 3] - Conecta entrada (3) à camada oculta (2)
- **b[1]**: [2 × 1] - Bias da camada oculta
- **W[2]**: [1 × 2] - Conecta camada oculta (2) à saída (1)
- **b[2]**: [1 × 1] - Bias da camada de saída

## 🔢 Equações Matemáticas

### Forward Pass

1. **Camada Oculta**:
   ```
   z[1] = W[1] @ x + b[1]
   a[1] = ReLU(z[1]) = max(0, z[1])
   ```

2. **Camada de Saída**:
   ```
   z[2] = W[2] @ a[1] + b[2]
   a[2] = sigmoid(z[2]) = 1 / (1 + e^(-z[2]))
   ŷ = a[2]
   ```

3. **Loss Function** (Binary Cross-Entropy):
   ```
   L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
   ```

### Backward Pass (Backpropagation)

Usa a **Chain Rule** para calcular os gradientes:

1. **Derivada da Loss**:
   ```
   ∂L/∂a[2] = a[2] - y
   ```

2. **Camada de Saída**:
   ```
   ∂a[2]/∂z[2] = sigmoid'(z[2]) = a[2] * (1 - a[2])
   ∂L/∂z[2] = ∂L/∂a[2] * ∂a[2]/∂z[2]
   
   ∂L/∂W[2] = ∂L/∂z[2] @ a[1]ᵀ
   ∂L/∂b[2] = ∂L/∂z[2]
   ```

3. **Camada Oculta**:
   ```
   ∂L/∂a[1] = W[2]ᵀ @ ∂L/∂z[2]
   ∂a[1]/∂z[1] = ReLU'(z[1]) = {1 se z[1] > 0, 0 caso contrário}
   ∂L/∂z[1] = ∂L/∂a[1] * ∂a[1]/∂z[1]
   
   ∂L/∂W[1] = ∂L/∂z[1] @ xᵀ
   ∂L/∂b[1] = ∂L/∂z[1]
   ```

4. **Atualização dos Pesos** (Gradient Descent):
   ```
   W_novo = W_antigo - η * ∂L/∂W
   b_novo = b_antigo - η * ∂L/∂b
   ```
   onde η é a taxa de aprendizado (learning rate).

## 🚀 Como Executar

### Requisitos

```bash
pip install numpy
```

### Execução

```bash
python mlp_manual.py
```

## 📊 Exemplo de Saída

O programa executará uma iteração completa de treinamento e exibirá:

1. **Forward Pass**: Valores intermediários de cada camada
2. **Loss**: Valor da função de perda
3. **Backward Pass**: Todos os gradientes calculados
4. **Atualização**: Pesos antes e depois da atualização

```
=====================================
FORWARD PASS - PROPAGAÇÃO PARA FRENTE
=====================================

Input x: [0.5 0.6 0.1]

--- CAMADA 1 (Input -> Hidden) ---
z[1] = W[1] @ x + b[1]
z[1] = [[0.2 0.3 0.1]
        [0.4 -0.2 0.5]] @ [0.5 0.6 0.1] + [0.1 0.2]
z[1] = [0.39 0.38]

a[1] = ReLU(z[1]) = [0.39 0.38]

--- CAMADA 2 (Hidden -> Output) ---
...
```

## 🔑 Conceitos-Chave

### 1. Forward Pass
Processo de calcular a saída da rede propagando os dados de entrada através das camadas, da esquerda para direita.

### 2. Backpropagation
Algoritmo para calcular os gradientes da função de perda em relação aos pesos, propagando o erro de volta através da rede, da direita para esquerda.

### 3. Chain Rule
Regra do cálculo que permite calcular derivadas de funções compostas:
```
∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
```

### 4. Funções de Ativação

- **ReLU**: `max(0, z)`
  - Simples e eficiente
  - Evita o problema de vanishing gradient
  - Derivada: 1 se z > 0, 0 caso contrário

- **Sigmoid**: `1 / (1 + e^(-z))`
  - Saída entre 0 e 1
  - Ideal para classificação binária
  - Derivada: σ(z) * (1 - σ(z))

### 5. Loss Function

**Binary Cross-Entropy**: Mede a diferença entre a distribuição de probabilidade prevista e a real para classificação binária.

## 🎓 Aplicação Prática

Este código é útil para:

1. ✅ **Entender backpropagation**: Ver cada passo do cálculo
2. ✅ **Verificar implementações**: Comparar com frameworks
3. ✅ **Estudar para provas**: Material didático com cálculos completos
4. ✅ **Debug**: Identificar problemas em redes neurais

## 📖 Referências

1. Stanford CS224N - Neural Networks and Neural Language Models
2. Rumelhart, Hinton, Williams (1986) - Learning representations by back-propagating errors
3. Goodfellow, Bengio, Courville - Deep Learning Book

## 🤝 Modificações Sugeridas

Para explorar mais:

1. **Adicionar mais camadas**: Criar um MLP de 3+ camadas
2. **Testar outras ativações**: tanh, Leaky ReLU, etc.
3. **Múltiplas iterações**: Loop de treinamento completo
4. **Diferentes loss functions**: MSE, MAE, etc.
5. **Mini-batches**: Processar múltiplos exemplos de uma vez
6. **Regularização**: L1, L2, Dropout


