# Modelo de Linguagem com Janelas Deslizantes

## 📋 Descrição do Projeto

Este projeto implementa um modelo de linguagem simples baseado em redes neurais feedforward usando a técnica de **janelas deslizantes** (sliding windows). O objetivo é analisar como o tamanho da janela de contexto afeta a perplexidade do modelo.

## 🎯 Objetivo da Questão 8

Implementar um modelo de linguagem neural feedforward que:
- Utiliza janelas deslizantes para capturar contexto
- Prediz a próxima palavra baseado nas N palavras anteriores
- Analisa como a perplexidade varia com diferentes tamanhos de janela

## 🏗️ Arquitetura do Modelo

### Componentes Principais

1. **Camada de Embedding**
   - Converte palavras em vetores densos de dimensão fixa
   - Permite capturar similaridades semânticas

2. **Camadas Feedforward**
   - Concatena os embeddings da janela de contexto
   - Camada oculta com ReLU e Dropout
   - Camada de saída com dimensão igual ao vocabulário

3. **Função de Perda**
   - Cross-Entropy Loss para classificação multiclasse
   - Cada palavra do vocabulário é uma classe

### Equações do Modelo

```
Entrada: [w₁, w₂, ..., wₙ] → Embedding → [e₁, e₂, ..., eₙ]

h = ReLU(W₁ · concat(e₁, e₂, ..., eₙ) + b₁)

y = softmax(W₂ · h + b₂)
```

Onde:
- `n` = tamanho da janela
- `eᵢ` = embedding da palavra i
- `h` = representação oculta
- `y` = distribuição de probabilidade sobre o vocabulário

## 📊 Janelas Deslizantes

### Como Funciona

Para o texto: "the cat sat on the mat"

Com janela de tamanho 3:
```
Janela 1: [the, cat, sat] → Prediz: on
Janela 2: [cat, sat, on] → Prediz: the
Janela 3: [sat, on, the] → Prediz: mat
```

### Vantagens
- ✅ Captura dependências locais
- ✅ Eficiente computacionalmente
- ✅ Simples de implementar

### Limitações
- ❌ Contexto fixo e limitado
- ❌ Não captura dependências de longo alcance
- ❌ Precisa de mais parâmetros para janelas maiores

## 📈 Perplexidade

### Definição

A perplexidade mede quão "surpreso" o modelo fica com os dados de teste:

```
Perplexidade = exp(- (1/N) Σ log P(wᵢ|contexto))
```

### Interpretação

- **Perplexidade baixa**: Modelo confiante e preciso
- **Perplexidade alta**: Modelo incerto ou impreciso
- **Perplexidade de N**: Equivalente a escolher aleatoriamente entre N palavras

### Exemplo

Se o modelo tem perplexidade 50:
- É como se estivesse escolhendo entre 50 palavras igualmente prováveis
- Quanto menor, melhor o modelo

## 🔬 Análise: Tamanho da Janela vs Perplexidade

### Experimento

O código treina modelos com janelas de tamanho 2, 3, 4, 5 e 6 palavras.

### Resultados Esperados

#### Janela Pequena (2-3 palavras)
- **Perplexidade**: Alta
- **Motivo**: Contexto insuficiente para boas predições
- **Exemplo**: "the cat" → difícil prever se é "sat", "ran", "jumped", etc.

#### Janela Média (4-5 palavras)
- **Perplexidade**: Redução significativa
- **Motivo**: Contexto suficiente para capturar padrões
- **Exemplo**: "the cat sat on the" → mais provável prever "mat" ou "floor"

#### Janela Grande (6+ palavras)
- **Perplexidade**: Diminui menos ou estabiliza
- **Motivo**: Overfitting ou dados insuficientes
- **Trade-off**: Mais parâmetros vs quantidade de dados

### Gráfico Típico

```
Perplexidade
    |
150 |  •
    |
120 |    •
    |
 90 |      •
    |
 60 |        •
    |
 30 |          • ___________
    |________________________
      2   3   4   5   6   → Tamanho da Janela
```

## 🚀 Como Executar

### Requisitos

```bash
pip install torch numpy matplotlib
```

### Execução

```bash
python language_model.py
```

### Saída

O programa irá:
1. Treinar modelos com diferentes tamanhos de janela
2. Imprimir a perplexidade ao longo do treinamento
3. Gerar gráficos comparativos
4. Salvar visualizações em `perplexity_analysis.png`

## 📁 Estrutura do Código

```
language_model.py
├── TextDataset          # Cria janelas deslizantes do texto
├── FeedForwardLM        # Arquitetura da rede neural
├── build_vocab          # Constrói vocabulário do texto
├── calculate_perplexity # Calcula métrica de avaliação
├── train_model          # Loop de treinamento
└── main                 # Experimentos e visualização
```

## 🔧 Hiperparâmetros

```python
embedding_dim = 64      # Dimensão dos embeddings
hidden_dim = 128        # Dimensão da camada oculta
batch_size = 32         # Tamanho do batch
epochs = 30             # Número de épocas
learning_rate = 0.001   # Taxa de aprendizado
dropout = 0.2           # Taxa de dropout
```

## 📝 Customização

### Usar seu próprio texto

Substitua a variável `sample_text` no código:

```python
with open('seu_texto.txt', 'r', encoding='utf-8') as f:
    sample_text = f.read()
```

### Testar outros tamanhos de janela

Modifique a lista:

```python
window_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
```

### Ajustar arquitetura

```python
# Modelo mais profundo
embedding_dim = 128
hidden_dim = 256

# Adicionar mais camadas no FeedForwardLM
self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
self.fc3 = nn.Linear(hidden_dim // 2, vocab_size)
```

## 📊 Interpretação dos Resultados

### Comparação com os Slides

Baseado nos slides de Stanford (Jurafsky & Martin):

1. **Modelos N-gram vs Neural**
   - N-grams: Contam ocorrências diretas
   - Neural: Aprendem representações contínuas

2. **Vantagens do Modelo Neural**
   - Generalização para palavras similares
   - Embeddings capturam semântica
   - Menos esparsidade

3. **Trade-offs**
   - Janela maior → Mais contexto → Menos perplexidade
   - Janela maior → Mais parâmetros → Risco de overfitting
   - Janela maior → Menos exemplos de treino

## 🎓 Conceitos Importantes

### 1. Problema de Esparsidade
- N-grams sofrem com combinações não vistas
- Redes neurais generalizam via embeddings

### 2. Maldição da Dimensionalidade
- Janelas maiores aumentam o espaço de busca
- Requer mais dados de treinamento

### 3. Limitações do Feedforward
- Contexto fixo (não como RNNs/Transformers)
- Não compartilha pesos entre posições
- Não processa sequências de tamanho variável

## 📚 Referências

1. Jurafsky & Martin - Speech and Language Processing
   - [Slides de Redes Neurais](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)
   - [Slides de Language Models](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf)

2. Bengio et al. (2003) - "A Neural Probabilistic Language Model"
   - Primeiro modelo neural de linguagem feedforward

3. Goodfellow et al. - Deep Learning Book
   - Capítulo sobre Sequence Modeling

## 💡 Próximos Passos

Para melhorar o modelo:

1. **RNN/LSTM**: Contexto variável e memória
2. **Transformers**: Atenção e paralelização
3. **Pre-training**: Transfer learning (BERT, GPT)
4. **Regularização**: Weight decay, early stopping
5. **Dados**: Corpus maior (ex: WikiText, Penn Treebank)

## 🤝 Conclusão

Este projeto demonstra:
- ✅ Implementação de modelo neural de linguagem
- ✅ Análise empírica de janelas deslizantes
- ✅ Relação entre tamanho de contexto e perplexidade
- ✅ Fundamentos para arquiteturas mais avançadas

**Resultado esperado**: A perplexidade geralmente diminui à medida que aumentamos o tamanho da janela, até um ponto de diminishing returns onde contextos muito grandes não trazem ganhos significativos (especialmente com dados limitados).

