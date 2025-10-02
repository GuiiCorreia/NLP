"""
Demonstração: Por que um Perceptron Simples não resolve XOR
e como uma Rede de Duas Camadas (MLP) consegue resolver

Autor: [Seu Nome]
Data: Outubro 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== DADOS DO XOR ====================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("="*60)
print("PROBLEMA DO XOR")
print("="*60)
print("\nTabela Verdade:")
print("X1 | X2 | Y (XOR)")
print("-" * 20)
for i in range(len(X)):
    print(f" {X[i][0]} |  {X[i][1]} | {y[i][0]}")
print()


# ==================== FUNÇÕES DE ATIVAÇÃO ====================
def sigmoid(x):
    """Função de ativação sigmoid"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow

def sigmoid_derivative(x):
    """Derivada da função sigmoid"""
    return x * (1 - x)

def step_function(x):
    """Função degrau para perceptron simples"""
    return (x > 0).astype(int)


# ==================== PERCEPTRON SIMPLES ====================
class SimplePerceptron:
    """
    Perceptron Simples (Uma Camada)
    Demonstra que não consegue aprender XOR
    """
    
    def __init__(self, input_size=2, learning_rate=0.1):
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.random.randn(1, 1)  # Corrigido para (1, 1)
        self.learning_rate = learning_rate
        
    def predict(self, X):
        """Faz predição usando função degrau"""
        z = np.dot(X, self.weights) + self.bias
        return step_function(z)
    
    def train(self, X, y, epochs=1000):
        """Treina o perceptron usando regra de aprendizado do perceptron"""
        history = []
        
        for epoch in range(epochs):
            total_error = 0
            
            for i in range(len(X)):
                # Forward pass
                prediction = self.predict(X[i:i+1])
                error = y[i:i+1] - prediction
                total_error += np.abs(error).sum()
                
                # Atualizar pesos (regra do perceptron)
                self.weights += self.learning_rate * X[i:i+1].T * error
                self.bias += self.learning_rate * error
            
            if epoch % 100 == 0:
                history.append(total_error)
        
        return history
    
    def evaluate(self, X, y):
        """Avalia acurácia do modelo"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy, predictions


# ==================== REDE DE DUAS CAMADAS (MLP) ====================
class TwoLayerNetwork:
    """
    Multi-Layer Perceptron (MLP) com uma camada oculta
    Consegue aprender XOR através de representações não-lineares
    """
    
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.5):
        # Inicializar pesos aleatórios com Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def forward(self, X):
        """Forward pass através da rede"""
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        """Backpropagation para calcular gradientes"""
        m = X.shape[0]
        
        # Gradiente da camada de saída
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Gradiente da camada oculta
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Atualizar pesos
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=5000):
        """Treina a rede usando backpropagation"""
        history = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calcular erro (MSE)
            loss = np.mean((y - output) ** 2)
            
            # Backward pass
            self.backward(X, y)
            
            if epoch % 500 == 0:
                history.append(loss)
                
        return history
    
    def predict(self, X):
        """Faz predição (output > 0.5 = 1, senão 0)"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Avalia acurácia do modelo"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy, predictions


# ==================== TREINAMENTO E RESULTADOS ====================
print("\n" + "="*60)
print("PARTE 1: PERCEPTRON SIMPLES (UMA CAMADA)")
print("="*60)

# Treinar perceptron simples
perceptron = SimplePerceptron(input_size=2, learning_rate=0.1)
print("\nTreinando perceptron simples...")
perceptron.train(X, y, epochs=1000)

# Avaliar
accuracy, predictions = perceptron.evaluate(X, y)
print(f"\nAcurácia: {accuracy * 100:.1f}%")
print("\nPredições:")
print("Entrada | Esperado | Predito | Correto?")
print("-" * 45)
for i in range(len(X)):
    correct = "✓" if predictions[i][0] == y[i][0] else "✗"
    print(f"  {X[i]}  |    {y[i][0]}     |    {predictions[i][0]}    |   {correct}")

print("\n⚠️  CONCLUSÃO: Perceptron simples NÃO consegue aprender XOR!")
print("Razão: XOR não é linearmente separável.")

print("\n" + "="*60)
print("PARTE 2: REDE DE DUAS CAMADAS (MLP)")
print("="*60)

# Treinar MLP
mlp = TwoLayerNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)
print("\nTreinando MLP (2 camadas)...")
history = mlp.train(X, y, epochs=5000)

# Avaliar
accuracy, predictions = mlp.evaluate(X, y)
print(f"\nAcurácia: {accuracy * 100:.1f}%")
print("\nPredições:")
print("Entrada | Esperado | Predito | Probabilidade | Correto?")
print("-" * 60)
for i in range(len(X)):
    prob = mlp.forward(X[i:i+1])[0][0]
    correct = "✓" if predictions[i][0] == y[i][0] else "✗"
    print(f"  {X[i]}  |    {y[i][0]}     |    {predictions[i][0]}    |     {prob:.4f}    |   {correct}")

print("\n✅ CONCLUSÃO: MLP consegue aprender XOR perfeitamente!")
print("Razão: Camada oculta cria representações não-lineares.")


# ==================== VISUALIZAÇÃO ====================
def plot_decision_boundary(model, model_name):
    """Plota a fronteira de decisão do modelo"""
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plotar pontos XOR
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)
    
    # Adicionar rótulos
    for i, (x1, x2) in enumerate(X):
        plt.annotate(f'({x1},{x2})→{y[i][0]}', 
                    xy=(x1, x2), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10)
    
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.title(f'Fronteira de Decisão - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    return plt

print("\n" + "="*60)
print("VISUALIZAÇÃO")
print("="*60)
print("\nGerando gráficos das fronteiras de decisão...")

# Plot perceptron simples
try:
    plot_decision_boundary(perceptron, "Perceptron Simples (FALHA)")
    plt.savefig('perceptron_simples.png', dpi=150, bbox_inches='tight')
    print("✓ Salvo: perceptron_simples.png")
    plt.close()
except Exception as e:
    print(f"⚠ Erro ao gerar gráfico do perceptron: {e}")

# Plot MLP
try:
    plot_decision_boundary(mlp, "MLP (SUCESSO)")
    plt.savefig('mlp_xor.png', dpi=150, bbox_inches='tight')
    print("✓ Salvo: mlp_xor.png")
    plt.close()
except Exception as e:
    print(f"⚠ Erro ao gerar gráfico do MLP: {e}")

# Plot de convergência
try:
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, 5000, 500), history, marker='o', linewidth=2, color='green')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Erro (MSE)', fontsize=12)
    plt.title('Convergência do MLP', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Arquitetura da rede
    plt.text(0.5, 0.9, 'Arquitetura do MLP', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.7, 'Input Layer: 2 neurônios', ha='center', fontsize=11)
    plt.text(0.5, 0.5, '↓', ha='center', fontsize=16)
    plt.text(0.5, 0.4, 'Hidden Layer: 2 neurônios (sigmoid)', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    plt.text(0.5, 0.2, '↓', ha='center', fontsize=16)
    plt.text(0.5, 0.1, 'Output Layer: 1 neurônio (sigmoid)', ha='center', fontsize=11)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('convergencia_arquitetura.png', dpi=150, bbox_inches='tight')
    print("✓ Salvo: convergencia_arquitetura.png")
    plt.close()
except Exception as e:
    print(f"⚠ Erro ao gerar gráfico de convergência: {e}")

print("\n" + "="*60)
print("EXPERIMENTO CONCLUÍDO!")
print("="*60)
print("\nArquivos gerados:")
print("  • perceptron_simples.png - Mostra falha do perceptron")
print("  • mlp_xor.png - Mostra sucesso do MLP")
print("  • convergencia_arquitetura.png - Convergência e arquitetura")
print("\n" + "="*60)