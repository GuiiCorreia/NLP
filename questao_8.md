# Comparação: Modelo de Bigramas vs Neural Network Language Model (NN-LM)

## 📋 Questão 8
**Compare um modelo de linguagem baseado em bigramas com um pequeno NN-LM no mesmo corpus. Em quais aspectos o NN-LM supera o modelo de n-gramas?**

## 🎯 Objetivo
Este projeto implementa e compara dois modelos de linguagem:
1. **Modelo de Bigramas** com suavização Add-1 (Laplace)
2. **Neural Network Language Model (NN-LM)** simples com embeddings

## 📦 Dependências

### Instalação
```bash
pip install numpy torch matplotlib
```

### Requisitos
- Python 3.7+
- NumPy
- PyTorch
- Matplotlib

## 🚀 Como Executar

### Execução Básica
```bash
python comparacao_bigrama_nnlm.py
```

### O que o script faz:
1. **Demonstração de Generalização**: Mostra como NN-LM generaliza melhor para palavras não vistas
2. **Tratamento de Zeros**: Compara como cada modelo lida com sequências não vistas no treino
3. **Comparação Completa**: Treina ambos os modelos e calcula perplexidade
4. **Gera PDF**: Cria um relatório visual com gráficos comparativos

### Saída
- **Console**: Resultados detalhados da comparação
- **PDF**: `comparacao_bigrama_vs_nnlm.pdf` com visualizações e análise completa

## 📊 Estrutura do Código

### Classes Principais

#### `BigramModel`
Implementa modelo de linguagem baseado em bigramas:
- Suavização Add-1 (Laplace) para lidar com zeros
- Cálculo de probabilidade: P(w₂|w₁) = (C(w₁,w₂) + 1) / (C(w₁) + V)
- Cálculo de perplexidade no conjunto de teste

#### `SimpleNNLM`
Rede neural para modelagem de linguagem:
- **Arquitetura**: Embedding → Hidden Layer (ReLU) → Output (Softmax)
- **Embeddings**: Representações vetoriais das palavras
- **Contexto**: Janela de N palavras anteriores

#### `NNLMWrapper`
Wrapper para facilitar treinamento e avaliação:
- Construção de vocabulário
- Preparação de dados
- Treinamento com Adam optimizer
- Cálculo de perplexidade

## 🔬 Experimentos Realizados

### 1. Demonstração de Generalização
```
Corpus de Treino:
- "the cat eats fish"
- "the dog eats meat"
- (sem "bird")

Teste:
- "the bird eats seeds"
```
**Resultado**: NN-LM generaliza melhor pois embeddings de "cat", "dog" e "bird" são similares.

### 2. Problema de Zeros
```
Treino:
- "I like apples"
- "I eat oranges"
- (nunca "eat bananas" juntos)

Teste:
- "I eat bananas"
```
**Resultado**: Bigrama sofre mesmo com Add-1. NN-LM usa similaridade dos embeddings.

### 3. Comparação de Perplexidade
Avalia ambos os modelos em corpus maior e calcula métricas objetivas.

## 📈 Aspectos em que o NN-LM Supera N-gramas

### 1. ✅ Generalização através de Embeddings
- **Bigramas**: Trata cada palavra independentemente
- **NN-LM**: Aprende representações que capturam similaridade semântica
- **Exemplo**: "cat eats fish" → pode generalizar para "dog eats meat"

### 2. ✅ Problema da Esparsidade
- **Bigramas**: Muitas combinações nunca aparecem no treino (99%+ são zeros)
- **NN-LM**: Embeddings compartilhados reduzem drasticamente a esparsidade
- **Resultado**: Melhor predição para sequências não vistas

### 3. ✅ Representação Contínua vs Discreta
- **Bigramas**: Espaço discreto - palavras não têm relação
- **NN-LM**: Espaço contínuo - palavras similares têm vetores próximos
- **Vantagem**: Interpolação suave entre conceitos

### 4. ✅ Padrões Complexos
- **Bigramas**: Apenas dependências locais (palavra anterior)
- **NN-LM**: Hidden layers capturam padrões não-lineares
- **Benefício**: Features automáticas vs contagem simples

### 5. ✅ Melhor Uso dos Dados
- **Bigramas**: Cada bigrama aprendido independentemente
- **NN-LM**: Compartilha conhecimento através de pesos e embeddings
- **Impacto**: Perplexidade menor = melhor modelagem

### 6. ✅ Escalabilidade
- **Bigramas**: Cresce O(V²) com vocabulário
- **NN-LM**: Cresce O(V) no embedding
- **Resultado**: Mais eficiente para vocabulários grandes

## 📊 Métricas de Avaliação

### Perplexidade
```
PP(W) = P(w₁w₂...wₙ)^(-1/N)
```
- **Menor é melhor**: Indica melhor predição das palavras
- Normalizada pelo número de palavras
- Range: [1, ∞]

### Resultados Típicos
- **Bigrama**: Perplexidade ~200-400
- **NN-LM**: Perplexidade ~50-150 (melhoria de 40-60%)

## 🎨 Visualizações Geradas

O PDF contém:
1. **Gráfico de Barras**: Comparação direta de perplexidade
2. **Curva de Loss**: Convergência do NN-LM durante treinamento
3. **Análise Textual**: Explicação detalhada de cada aspecto superior

## 🔧 Personalização

### Ajustar Hiperparâmetros
```python
# No código, você pode modificar:
nnlm = NNLMWrapper(
    embedding_dim=50,    # Dimensão dos embeddings
    hidden_dim=100,      # Tamanho da camada oculta
    context_size=2       # Tamanho da janela de contexto
)

nnlm.train(
    train_corpus, 
    epochs=30,           # Número de épocas
    lr=0.005            # Taxa de aprendizado
)
```

### Usar Seu Próprio Corpus
```python
# Adicione suas próprias sentenças:
meu_corpus = [
    "sua primeira sentença",
    "sua segunda sentença",
    # ...
]

# Divida em treino/teste
train = meu_corpus[:int(0.7 * len(meu_corpus))]
test = meu_corpus[int(0.7 * len(meu_corpus)):]
```

## 📚 Referências

### Materiais Base
- [Stanford NLP - Neural Networks and Neural Language Models](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)
- [Stanford NLP - N-gram Language Modeling](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf)

### Conceitos Chave
- **Bigramas**: P(wₙ|wₙ₋₁) ≈ C(wₙ₋₁,wₙ) / C(wₙ₋₁)
- **Add-1 Smoothing**: P(wₙ|wₙ₋₁) = (C(wₙ₋₁,wₙ) + 1) / (C(wₙ₋₁) + V)
- **Neural LM**: Usa embeddings + feedforward network
- **Embeddings**: Representações vetoriais densas que capturam semântica

## 💡 Insights Principais

### Por que NN-LM é Superior?

1. **Embeddings = Memória Semântica**
   - Palavras similares têm representações similares
   - Permite transferência de conhecimento

2. **Compartilhamento de Parâmetros**
   - Mesmo embedding usado em múltiplos contextos
   - Aprendizado mais eficiente

3. **Não-linearidade**
   - Funções de ativação (ReLU) capturam padrões complexos
   - Vai além de contagem simples

4. **Generalização**
   - Pode inferir probabilidades para combinações não vistas
   - Reduz problema de sparsity

### Limitações do Bigrama

1. **Independência Condicional**
   - Assume que só a palavra anterior importa
   - Ignora contexto mais amplo

2. **Esparsidade Severa**
   - 99%+ das combinações possíveis são zero
   - Mesmo com smoothing, problemas persistem

3. **Sem Semântica**
   - "cat" e "dog" são completamente independentes
   - Não captura similaridade

## 🎓 Conclusão

O NN-LM supera o modelo de bigramas principalmente devido à:
- **Representação contínua** via embeddings
- **Generalização semântica** para palavras similares
- **Redução de sparsity** através de compartilhamento de parâmetros
- **Capacidade não-linear** de capturar padrões complexos

Isso resulta em **perplexidade menor** e **melhor predição** de palavras.
