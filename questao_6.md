# Implementação de Mean e Max Pooling para Embeddings

## 📝 Descrição da Questão

Dada uma matriz de embeddings de uma frase, implementar mean pooling e max pooling, e analisar qual das estratégias preserva mais informação no exemplo.

## 🎯 Objetivo

Este projeto implementa e compara duas estratégias fundamentais de pooling usadas em Processamento de Linguagem Natural (PLN) para agregar embeddings de sequências:

- **Mean Pooling**: Calcula a média de cada dimensão ao longo da sequência
- **Max Pooling**: Seleciona o valor máximo de cada dimensão ao longo da sequência

## 🚀 Como Executar

### Requisitos
```bash
pip install numpy matplotlib seaborn
```

### Execução
```bash
python pooling_implementation.py
```

## 📊 O que o código faz

1. **Cria uma matriz de embeddings simulada** (5 tokens × 8 dimensões)
2. **Aplica mean pooling e max pooling**
3. **Calcula métricas de preservação de informação**
4. **Gera visualizações comparativas** (salvas como `pooling_comparison.png`)
5. **Apresenta análise detalhada** de qual método preserva mais informação

## 🔍 Estratégias de Pooling

### Mean Pooling

**Fórmula**: Para cada dimensão `d`:
```
result[d] = (1/n) × Σ embeddings[i][d]
```

**Características**:
- ✅ Preserva a distribuição geral dos valores
- ✅ Robusta a outliers
- ✅ Todos os tokens contribuem igualmente
- ❌ Pode diluir informações importantes
- ❌ Perde valores extremos

**Quando usar**: 
- Quando todos os tokens são igualmente importantes
- Para representações balanceadas de documentos
- Em classificação de sentimentos onde o contexto geral importa

### Max Pooling

**Fórmula**: Para cada dimensão `d`:
```
result[d] = max(embeddings[:][d])
```

**Características**:
- ✅ Preserva características salientes
- ✅ Captura as features mais ativadas
- ✅ Útil para detecção de padrões específicos
- ❌ Ignora valores médios e baixos
- ❌ Sensível a outliers

**Quando usar**:
- Quando features específicas são críticas
- Em detecção de entidades ou keywords
- Para capturar sinais fortes em qualquer posição

## 📈 Análise de Preservação de Informação

### Métricas Utilizadas

1. **Variância**: Mede a dispersão dos valores
2. **Norma**: Magnitude do vetor resultante
3. **Range**: Amplitude dos valores (max - min)
4. **Distribuição**: Comparação visual das distribuições

### Resultados do Exemplo

No exemplo implementado (com embeddings simulados):

- **Max Pooling** tende a preservar maior variância e valores extremos
- **Mean Pooling** tende a preservar melhor a distribuição geral

## 🎓 Conceitos Teóricos

### Por que usar Pooling?

Embeddings de sequências têm dimensão `(sequence_length, embedding_dim)`, mas muitas tarefas precisam de vetores de tamanho fixo `(embedding_dim)`. Pooling resolve isso agregando a informação da sequência.

### Relação com o Material (Jurafsky & Martin)

Conforme os slides de Stanford, pooling é uma operação crucial em:
- Redes neurais para classificação de texto
- Sentence embeddings
- Redução de dimensionalidade temporal

## 🏆 Conclusão

**Qual preserva mais informação?**

A resposta depende do tipo de informação que é valiosa para a tarefa:

### Mean Pooling preserva:
- Informação sobre a **distribuição geral**
- Tendências centrais
- Contribuições balanceadas de todos os tokens

### Max Pooling preserva:
- Informação sobre **características salientes**
- Valores extremos (picos de ativação)
- Features mais discriminativas

**Na prática**: Mean pooling é mais comum em modelos modernos (como sentence-BERT) porque fornece representações mais estáveis e balanceadas. Max pooling é preferido quando características específicas são mais importantes que o contexto geral.

## 📚 Referências

- Jurafsky & Martin - Speech and Language Processing
- [Stanford NLP Slides](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)
- Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
- Reimers & Gurevych (2019) - Sentence-BERT

## 👨‍💻 Estrutura do Código

```
pooling_implementation.py
├── mean_pooling()          # Implementa mean pooling
├── max_pooling()           # Implementa max pooling
├── calcular_preservacao_informacao()  # Métricas
├── visualizar_comparacao() # Gera gráficos
└── main()                  # Execução principal
```

## 🖼️ Visualizações Geradas

O script gera um arquivo `pooling_comparison.png` com 4 subplots:
1. Heatmap da matriz de embeddings original
2. Comparação direta dos vetores resultantes
3. Distribuição dos valores
4. Análise por dimensão

## 💡 Experimente

Você pode modificar a matriz de embeddings em `main()` para testar com:
- Diferentes tamanhos de sequência
- Diferentes dimensões de embedding
- Padrões específicos (valores esparsos, densos, etc.)

