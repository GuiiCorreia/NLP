import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Implementação do Perceptron para classificação binária.
    
    Baseado no material: Um perceptron é uma unidade neural simples
    com saída binária (0 ou 1) e sem função de ativação não-linear.
    
    Regra de decisão:
    y = 0 se w·x + b ≤ 0
    y = 1 se w·x + b > 0
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_history = []
        
    def fit(self, X, y):
        """
        Treina o perceptron usando o algoritmo de aprendizado.
        
        Args:
            X: array (n_samples, n_features) - dados de entrada
            y: array (n_samples,) - labels (0 ou 1)
        """
        n_samples, n_features = X.shape
        
        # Inicializa pesos e bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Treinamento
        for iteration in range(self.n_iterations):
            errors = 0
            
            for idx, x_i in enumerate(X):
                # Calcula a soma ponderada: z = w·x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Aplica função de ativação (step function)
                y_predicted = 1 if linear_output > 0 else 0
                
                # Atualiza pesos se houver erro
                error = y[idx] - y_predicted
                if error != 0:
                    # Regra de atualização do perceptron:
                    # w = w + learning_rate * (y_true - y_pred) * x
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors += 1
            
            self.errors_history.append(errors)
            
            # Para se não houver mais erros
            if errors == 0:
                print(f"Convergiu na iteração {iteration + 1}")
                break
                
        return self
    
    def predict(self, X):
        """Prediz as classes para novos dados."""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output > 0, 1, 0)
    
    def get_decision_boundary_equation(self):
        """
        Retorna a equação da reta de decisão.
        
        A fronteira de decisão ocorre quando w·x + b = 0
        Para 2D: w1*x1 + w2*x2 + b = 0
        Forma padrão: x2 = (-w1/w2)*x1 + (-b/w2)
        """
        w1, w2 = self.weights
        b = self.bias
        
        if w2 == 0:
            return f"x1 = {-b/w1:.4f} (linha vertical)"
        
        slope = -w1 / w2
        intercept = -b / w2
        
        return {
            'equacao_geral': f"{w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0",
            'forma_padrao': f"x2 = {slope:.4f}*x1 + {intercept:.4f}",
            'slope': slope,
            'intercept': intercept,
            'weights': self.weights,
            'bias': self.bias
        }


def generate_linearly_separable_data(n_samples=100, random_state=42):
    """
    Gera dados linearmente separáveis para duas classes.
    """
    np.random.seed(random_state)
    
    # Classe 0: pontos em torno de (2, 2)
    X_class0 = np.random.randn(n_samples//2, 2) * 0.8 + [2, 2]
    y_class0 = np.zeros(n_samples//2)
    
    # Classe 1: pontos em torno de (5, 5)
    X_class1 = np.random.randn(n_samples//2, 2) * 0.8 + [5, 5]
    y_class1 = np.ones(n_samples//2)
    
    # Combina as classes
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Embaralha
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def plot_results(X, y, perceptron):
    """Visualiza os dados e a reta de decisão aprendida."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Dados e reta de decisão
    plt.subplot(1, 2, 1)
    
    # Plota os pontos
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], 
                c='blue', marker='o', label='Classe 0', s=100, alpha=0.7)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], 
                c='red', marker='s', label='Classe 1', s=100, alpha=0.7)
    
    # Plota a reta de decisão
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    eq = perceptron.get_decision_boundary_equation()
    
    if isinstance(eq, str):  # linha vertical
        x1_val = float(eq.split('=')[1].strip().split()[0])
        plt.axvline(x=x1_val, color='green', linestyle='--', 
                   linewidth=2, label='Fronteira de Decisão')
    else:
        x1_line = np.linspace(x1_min, x1_max, 100)
        x2_line = eq['slope'] * x1_line + eq['intercept']
        plt.plot(x1_line, x2_line, 'g--', linewidth=2, 
                label='Fronteira de Decisão')
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Perceptron - Classificação Linear', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Convergência
    plt.subplot(1, 2, 2)
    plt.plot(perceptron.errors_history, 'b-', linewidth=2)
    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Número de Erros', fontsize=12)
    plt.title('Convergência do Perceptron', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

print("="*70)
print("QUESTÃO 1: PERCEPTRON PARA CLASSIFICAÇÃO BINÁRIA EM 2D")
print("="*70)

# Gera dados linearmente separáveis
X, y = generate_linearly_separable_data(n_samples=100)
print(f"\n✓ Dados gerados: {len(X)} pontos, {np.sum(y==0)} da classe 0, {np.sum(y==1)} da classe 1")

# Treina o perceptron
perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X, y)

# Avalia o desempenho
y_pred = perceptron.predict(X)
accuracy = np.mean(y_pred == y) * 100
print(f"✓ Acurácia no conjunto de treinamento: {accuracy:.2f}%")

# Mostra a equação da reta de decisão
print("\n" + "="*70)
print("EQUAÇÃO DA RETA DE DECISÃO APRENDIDA:")
print("="*70)

eq = perceptron.get_decision_boundary_equation()
if isinstance(eq, dict):
    print(f"\n📐 Forma Geral: {eq['equacao_geral']}")
    print(f"📐 Forma Padrão: {eq['forma_padrao']}")
    print(f"\n   Coeficientes:")
    print(f"   • Peso w₁ = {eq['weights'][0]:.4f}")
    print(f"   • Peso w₂ = {eq['weights'][1]:.4f}")
    print(f"   • Bias b  = {eq['bias']:.4f}")
    print(f"\n   • Inclinação (slope) = {eq['slope']:.4f}")
    print(f"   • Intercepto y = {eq['intercept']:.4f}")
else:
    print(f"\n📐 {eq}")

print("\n" + "="*70)
print("INTERPRETAÇÃO:")
print("="*70)
print("""
A reta de decisão w₁*x₁ + w₂*x₂ + b = 0 separa o espaço em duas regiões:
  • Região 1 (w·x + b > 0):  Classe 1 (vermelho)
  • Região 0 (w·x + b ≤ 0):  Classe 0 (azul)

O perceptron aprendeu os pesos ótimos através do algoritmo iterativo que
ajusta w e b sempre que há erro de classificação.
""")

# Visualiza os resultados
plot_results(X, y, perceptron)

print("\n✓ Implementação completa do Perceptron finalizada!")
