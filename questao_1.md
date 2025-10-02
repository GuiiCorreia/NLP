# Implementação de Perceptron para Classificação Binária

Este projeto apresenta uma implementação do algoritmo Perceptron do zero, utilizando Python e NumPy, para resolver um problema de classificação binária em um espaço 2D. O código gera dados linearmente separáveis, treina o modelo Perceptron e, ao final, exibe a equação da reta de decisão aprendida, juntamente com uma visualização gráfica.

## O Problema

O objetivo deste projeto é responder à seguinte questão:

> Dado um conjunto de pontos em 2D, implemente um perceptron para classificar duas classes linearmente separáveis. Qual é a equação da reta de decisão aprendida?

## Como Executar

### Pré-requisitos

Certifique-se de ter Python instalado. Você precisará das seguintes bibliotecas:

-   NumPy
-   Matplotlib

Você pode instalá-las via pip:

```bash
pip install numpy matplotlib
```

### Execução

1.  Salve o código em um arquivo, por exemplo, `perceptron_classifier.py`.
2.  Execute o script a partir do seu terminal:

```bash
python perceptron_classifier.py
```

## Fundamentos Teóricos do Perceptron

Um Perceptron é a unidade neural mais simples e serve como base para redes neurais mais complexas. Suas características principais são:

-   **Saída binária**: A saída do neurônio é 0 ou 1.
-   **Classificador Linear**: Não utiliza uma função de ativação não-linear.

#### Regra de Decisão

A classificação de uma nova amostra **x** é feita com base na seguinte regra, onde **w** são os pesos e **b** é o bias:

$$
y = \begin{cases} 
1 & \text{se } \mathbf{w} \cdot \mathbf{x} + b > 0 \\
0 & \text{se } \mathbf{w} \cdot \mathbf{x} + b \le 0 
\end{cases}
$$

#### A Fronteira de Decisão

A fronteira de decisão do Perceptron é a linha (ou hiperplano, em mais dimensões) que separa as duas classes. Ela ocorre exatamente onde a entrada da função de ativação é zero:

$$w_1x_1 + w_2x_2 + b = 0$$

Essa equação pode ser reescrita na forma padrão de uma reta ($y = mx + c$), onde $x_2$ representa o eixo y e $x_1$ o eixo x:

$$x_2 = \left(-\frac{w_1}{w_2}\right)x_1 + \left(-\frac{b}{w_2}\right)$$

Onde:
-   **Inclinação (Slope)** = $-w_1/w_2$
-   **Intercepto (Intercept)** = $-b/w_2$

#### Algoritmo de Treinamento

O objetivo do treinamento é encontrar os valores de **w** e **b** que separam os dados corretamente. O algoritmo implementado segue estes passos:
1.  Inicializa os pesos **w** e o bias **b** (geralmente com zeros).
2.  Para cada exemplo de treinamento $(\mathbf{x}, y)$:
    a. Calcula a saída linear: $z = \mathbf{w} \cdot \mathbf{x} + b$.
    b. Prevê a classe $y'$ usando a regra de decisão.
    c. Se a previsão $y'$ estiver errada ($y' \neq y$):
        - Atualiza os pesos: $\mathbf{w} \leftarrow \mathbf{w} + \eta(y - y')\mathbf{x}$
        - Atualiza o bias: $b \leftarrow b + \eta(y - y')$
3.  Repete o passo 2 por um número definido de iterações ou até que não hajam mais erros.

## Análise da Implementação

O script fornecido materializa esses conceitos da seguinte forma:
-   A classe **`Perceptron`** encapsula toda a lógica, com o método **`fit()`** implementando o algoritmo de treinamento.
-   A função **`get_decision_boundary_equation()`** extrai os pesos e o bias aprendidos para calcular e exibir a equação da reta de decisão.
-   A função **`plot_results()`** visualiza os dados e a fronteira de decisão, provando graficamente que o Perceptron encontrou uma solução linear para o problema.

## Exemplo de Saída

Ao executar o script, a saída no terminal será semelhante a esta (os valores exatos podem variar):

```
======================================================================
QUESTÃO 1: PERCEPTRON PARA CLASSIFICAÇÃO BINÁRIA EM 2D
======================================================================

✓ Dados gerados: 100 pontos, 50 da classe 0, 50 da classe 1
Convergiu na iteração 4
✓ Acurácia no conjunto de treinamento: 100.00%

======================================================================
EQUAÇÃO DA RETA DE DECISÃO APRENDIDA:
======================================================================

📐 Forma Geral: 1.2583*x1 + 1.6311*x2 + -10.0211 = 0
📐 Forma Padrão: x2 = -0.7715*x1 + 6.1434

    Coeficientes:
    • Peso w₁ = 1.2583
    • Peso w₂ = 1.6311
    • Bias b  = -10.0211

    • Inclinação (slope) = -0.7715
    • Intercepto y = 6.1434
```
Adicionalmente, uma janela será exibida com os gráficos da classificação e da convergência do erro.
