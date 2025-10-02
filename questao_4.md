# Questão 4: Word Embeddings - Explicação e Implementação

## 📚 Introdução

Este documento explica e implementa **embeddings iniciais para palavras**, além de discutir como embeddings diferem de features construídas manualmente.

---

## 🎯 O que são Word Embeddings?

**Word embeddings** são representações vetoriais densas de palavras em um espaço contínuo de dimensão fixa. Cada palavra é mapeada para um vetor de números reais, onde palavras com significados similares têm representações vetoriais próximas.

### Características principais:

- **Dimensão fixa**: Geralmente entre 50-300 dimensões (independente do tamanho do vocabulário)
- **Vetores densos**: Todos os valores são significativos (não esparsos)
- **Aprendidos automaticamente**: Extraídos dos dados durante o treinamento
- **Capturam semântica**: Palavras similares têm vetores similares

### Exemplo conceitual:

```
Vocabulário: ["rei", "rainha", "homem", "mulher"]
Dimensão dos embeddings: 3

rei    → [0.8, 0.3, 0.1]
rainha → [0.7, 0.4, 0.2]
homem  → [0.6, 0.1, 0.1]
mulher → [0.5, 0.2, 0.2]
```

---

## 🔧 Inicialização de Embeddings

Os embeddings são tipicamente inicializados aleatoriamente e depois refinados durante o treinamento. Três métodos comuns:

### 1. **Inicialização Aleatória Uniforme**
```python
embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
```
- Valores entre -1 e 1
- Simples e eficaz

### 2. **Inicialização Xavier/Glorot**
```python
limit = np.sqrt(6 / (vocab_size + embedding_dim))
embeddings = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
```
- Mantém variância consistente
- Boa para treinamento com gradiente

### 3. **Inicialização Normal**
```python
embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
```
- Distribuição gaussiana
- Desvio padrão pequeno (0.01 - 0.1)

---

## 🆚 Embeddings vs Features Manuais

### **Features Manuais (One-Hot Encoding)**

One-hot encoding é a representação tradicional de palavras:

```
Vocabulário: ["gato", "cachorro", "carro"]

gato     → [1, 0, 0]
cachorro → [0, 1, 0]
carro    → [0, 0, 1]
```

#### Problemas:

| Problema | Descrição |
|----------|-----------|
| **Alta dimensionalidade** | Dimensão = tamanho do vocabulário (pode ser 10.000+) |
| **Esparsidade** | Apenas um valor é 1, todos outros são 0 |
| **Sem semântica** | Todas palavras são ortogonais (similaridade = 0) |
| **Não generaliza** | Não captura relações entre palavras |
| **Crescimento** | Cresce linearmente com novo vocabulário |

#### Exemplo de problema semântico:

```python
similaridade("gato", "cachorro") = 0.0  # Ambos são animais, mas one-hot não captura isso!
similaridade("gato", "carro")    = 0.0  # Mesmo valor, apesar de serem completamente diferentes
```

---

### **Word Embeddings**

Representação densa e contínua:

```
Vocabulário: ["gato", "cachorro", "carro"]
Dimensão: 3

gato     → [0.8, 0.7, 0.1]  (após treinamento)
cachorro → [0.7, 0.8, 0.2]  (similar a gato)
carro    → [0.1, 0.2, 0.9]  (diferente)
```

#### Vantagens:

| Vantagem | Descrição |
|----------|-----------|
| **Dimensionalidade fixa** | Sempre mesmo tamanho (ex: 100), não importa o vocabulário |
| **Densidade** | Todos valores são significativos |
| **Semântica** | Palavras similares têm vetores similares |
| **Generalização** | Transfere conhecimento entre palavras similares |
| **Operações algébricas** | Permite aritmética vetorial (ex: rei - homem + mulher ≈ rainha) |

#### Exemplo de captura semântica:

```python
similaridade("gato", "cachorro") = 0.87  # Alto! Ambos são animais
similaridade("gato", "carro")    = 0.23  # Baixo! Conceitos diferentes
```

---

## 📊 Tabela Comparativa

| Aspecto | One-Hot (Manual) | Embeddings |
|---------|------------------|------------|
| **Dimensão** | = tamanho vocabulário | Fixa (50-300) |
| **Esparsidade** | Esparso (99%+ zeros) | Denso (todos valores significativos) |
| **Semântica** | Não captura | Captura relações |
| **Similaridade** | Sempre 0 ou 1 | Valor contínuo [0, 1] |
| **Aprendizado** | Fixo, pré-definido | Aprendido dos dados |
| **Generalização** | Fraca | Forte |
| **Memória** | Alta (vocab_size × vocab_size) | Baixa (vocab_size × embedding_dim) |
| **Escalabilidade** | Ruim (cresce com vocab) | Boa (dimensão fixa) |

---

## 💡 Por que Embeddings são Melhores?

### 1. **Eficiência Computacional**
- Vocabulário de 10.000 palavras:
  - One-hot: 10.000 dimensões
  - Embeddings: 100 dimensões (100x menor!)

### 2. **Captura de Relações Semânticas**
```
Após treinamento:
vec("Paris") - vec("França") ≈ vec("Roma") - vec("Itália")
vec("rei") - vec("homem") ≈ vec("rainha") - vec("mulher")
```

### 3. **Generalização**
- Modelo aprende que "gato" e "felino" são similares
- Mesmo que "felino" apareça pouco nos dados de treino

### 4. **Transferência de Aprendizado**
- Embeddings pré-treinados (Word2Vec, GloVe, FastText)
- Reutilizáveis em múltiplas tarefas

---

## 🔬 Como Embeddings são Aprendidos?

Embeddings são aprendidos através de objetivos de treinamento:

### 1. **Word2Vec (Skip-gram)**
- Dada uma palavra, prediz palavras do contexto
- Exemplo: "o __gato__ está feliz" → prediz ["o", "está", "feliz"]

### 2. **Word2Vec (CBOW)**
- Dado contexto, prediz palavra central
- Exemplo: ["o", "está", "feliz"] → prediz "gato"

### 3. **GloVe**
- Baseado em co-ocorrências globais de palavras

### 4. **Embeddings Contextuais (BERT, GPT)**
- Embeddings variam com contexto
- "banco" (assento) vs "banco" (financeiro)

---

## 🚀 Como Usar a Implementação

### Instalação

```bash
pip install numpy matplotlib scikit-learn
```

### Executar

```bash
python word_embeddings.py
```

### Saídas

1. **Comparação detalhada** entre one-hot e embeddings
2. **Exemplos práticos** com vocabulário realista
3. **Visualização 2D** dos embeddings (`embeddings_visualization.png`)
4. **Demonstração de similaridade** entre palavras

---

## 📈 Visualização dos Embeddings

A implementação gera um gráfico 2D usando PCA (Principal Component Analysis) para visualizar os embeddings em duas dimensões:

- Palavras relacionadas aparecem próximas
- Clusters semânticos emergem naturalmente
- Facilita interpretação e debugging

---

## 🎓 Conceitos Importantes

### **Matriz de Embeddings**

```
Shape: (vocab_size, embedding_dim)

         dim_0  dim_1  dim_2  ...  dim_n
palavra_0  0.23   0.45  -0.12  ...   0.67
palavra_1  0.34  -0.23   0.56  ...  -0.45
palavra_2 -0.12   0.78   0.23  ...   0.34
...
```

### **Lookup de Embedding**

```python
# Para obter embedding de uma palavra:
word_idx = word2idx["gato"]  # Obter índice
embedding = embeddings[word_idx]  # Lookup na matriz
```

### **Similaridade de Cosseno**

Medida de similaridade entre dois vetores:

```
cos(A, B) = (A · B) / (||A|| × ||B||)

Valores:
  1.0  = idênticos
  0.0  = ortogonais (não relacionados)
 -1.0  = opostos
```

---

## 📝 Conclusão

### Principais Takeaways:

1. **Embeddings superam features manuais** por capturarem semântica
2. **Inicialização** é aleatória, refinamento vem do treinamento
3. **Dimensionalidade fixa** torna embeddings escaláveis
4. **Aprendizado automático** elimina necessidade de feature engineering
5. **Representação densa** é mais eficiente que esparsa

### Quando usar cada um:

- **One-Hot**: Vocabulários muito pequenos, problemas categóricos simples
- **Embeddings**: NLP moderno, grandes vocabulários, necessidade de semântica

---

## 📚 Referências

- [Stanford NLP - Neural Networks](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)
- [Stanford NLP - Vector Semantics](https://web.stanford.edu/~jurafsky/slp3/slides/vector25aug.pdf)
- Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
- Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"

---

## 🛠️ Estrutura do Código

```
word_embeddings.py
├── WordEmbeddings          # Classe principal de embeddings
│   ├── initialize_embeddings()  # Diferentes métodos de inicialização
│   ├── get_embedding()          # Recuperar embedding de palavra
│   ├── similarity()             # Calcular similaridade
│   └── visualize_embeddings_2d()  # Plotar em 2D
│
├── ManualFeatures          # Classe para one-hot encoding
│   └── one_hot_encode()    # Criar representação one-hot
│
├── compare_embeddings_vs_manual()  # Comparação detalhada
└── exemplo_pratico()       # Demonstração completa
```

