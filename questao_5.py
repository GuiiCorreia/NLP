import numpy as np

"""
Multi-Layer Perceptron (MLP) de 2 Camadas
Implementação Manual de Forward Pass e Backpropagation

Arquitetura:
- Camada de entrada: 3 neurônios (x1, x2, x3)
- Camada oculta: 2 neurônios com ativação ReLU
- Camada de saída: 1 neurônio com ativação sigmoid
- Loss: Binary Cross-Entropy
"""

class MLPManual:
    def __init__(self):
        """Inicializa os pesos da rede com valores específicos"""
        # Pesos da camada 1 (input -> hidden)
        # Dimensão: [n_hidden, n_input] = [2, 3]
        self.W1 = np.array([
            [0.2, 0.3, 0.1],  # pesos para neurônio oculto 1
            [0.4, -0.2, 0.5]  # pesos para neurônio oculto 2
        ])
        
        # Bias da camada 1
        self.b1 = np.array([0.1, 0.2])
        
        # Pesos da camada 2 (hidden -> output)
        # Dimensão: [n_output, n_hidden] = [1, 2]
        self.W2 = np.array([[0.6, 0.3]])
        
        # Bias da camada 2
        self.b2 = np.array([0.5])
        
        # Taxa de aprendizado
        self.learning_rate = 0.1
        
    def relu(self, z):
        """
        Função de ativação ReLU
        ReLU(z) = max(0, z)
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivada da ReLU
        dReLU/dz = 1 se z > 0, 0 caso contrário
        """
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """
        Função de ativação Sigmoid
        σ(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """
        Derivada da sigmoid
        dσ/dz = σ(z) * (1 - σ(z))
        """
        return a * (1 - a)
    
    def binary_cross_entropy(self, y_pred, y_true):
        """
        Loss: Binary Cross-Entropy
        L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
        """
        epsilon = 1e-10  # para evitar log(0)
        return -np.mean(y_true * np.log(y_pred + epsilon) + 
                       (1 - y_true) * np.log(1 - y_pred + epsilon))
    
    def forward_pass(self, x, verbose=True):
        """
        Forward Pass - Propagação para frente
        
        Etapas:
        1. z[1] = W[1] @ x + b[1]
        2. a[1] = ReLU(z[1])
        3. z[2] = W[2] @ a[1] + b[2]
        4. a[2] = sigmoid(z[2])
        5. ŷ = a[2]
        """
        if verbose:
            print("\n" + "="*60)
            print("FORWARD PASS - PROPAGAÇÃO PARA FRENTE")
            print("="*60)
            print(f"\nInput x: {x}")
        
        # Camada 1: Input -> Hidden
        self.z1 = np.dot(self.W1, x) + self.b1
        if verbose:
            print(f"\n--- CAMADA 1 (Input -> Hidden) ---")
            print(f"z[1] = W[1] @ x + b[1]")
            print(f"z[1] = {self.W1} @ {x} + {self.b1}")
            print(f"z[1] = {self.z1}")
        
        self.a1 = self.relu(self.z1)
        if verbose:
            print(f"\na[1] = ReLU(z[1]) = {self.a1}")
        
        # Camada 2: Hidden -> Output
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        if verbose:
            print(f"\n--- CAMADA 2 (Hidden -> Output) ---")
            print(f"z[2] = W[2] @ a[1] + b[2]")
            print(f"z[2] = {self.W2} @ {self.a1} + {self.b2}")
            print(f"z[2] = {self.z2}")
        
        self.a2 = self.sigmoid(self.z2)
        if verbose:
            print(f"\na[2] = sigmoid(z[2]) = {self.a2}")
            print(f"ŷ (predição) = {self.a2[0]:.6f}")
        
        return self.a2
    
    def backward_pass(self, x, y_true, verbose=True):
        """
        Backward Pass - Backpropagation
        
        Usa a Chain Rule para calcular os gradientes:
        ∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
        
        Etapas:
        1. Calcular ∂L/∂a[2] (derivada da loss em relação à saída)
        2. Calcular ∂L/∂z[2] = ∂L/∂a[2] * ∂a[2]/∂z[2]
        3. Calcular ∂L/∂W[2] e ∂L/∂b[2]
        4. Calcular ∂L/∂a[1] (propagando o erro para trás)
        5. Calcular ∂L/∂z[1] = ∂L/∂a[1] * ∂a[1]/∂z[1]
        6. Calcular ∂L/∂W[1] e ∂L/∂b[1]
        """
        if verbose:
            print("\n" + "="*60)
            print("BACKWARD PASS - BACKPROPAGATION")
            print("="*60)
        
        # Derivada da loss em relação à saída
        # Para binary cross-entropy: ∂L/∂a[2] = a[2] - y
        dL_da2 = self.a2 - y_true
        if verbose:
            print(f"\n--- CAMADA DE SAÍDA ---")
            print(f"∂L/∂a[2] = a[2] - y = {self.a2} - {y_true} = {dL_da2}")
        
        # Derivada em relação a z[2]
        # ∂L/∂z[2] = ∂L/∂a[2] * ∂a[2]/∂z[2]
        # onde ∂a[2]/∂z[2] = sigmoid'(z[2]) = a[2] * (1 - a[2])
        da2_dz2 = self.sigmoid_derivative(self.a2)
        dL_dz2 = dL_da2 * da2_dz2
        if verbose:
            print(f"\n∂a[2]/∂z[2] = sigmoid'(z[2]) = a[2]*(1-a[2]) = {da2_dz2}")
            print(f"∂L/∂z[2] = ∂L/∂a[2] * ∂a[2]/∂z[2] = {dL_dz2}")
        
        # Gradientes para W[2] e b[2]
        # ∂L/∂W[2] = ∂L/∂z[2] * ∂z[2]/∂W[2] = ∂L/∂z[2] @ a[1].T
        dL_dW2 = np.outer(dL_dz2, self.a1)
        dL_db2 = dL_dz2
        if verbose:
            print(f"\n∂L/∂W[2] = ∂L/∂z[2] @ a[1].T = {dL_dW2}")
            print(f"∂L/∂b[2] = ∂L/∂z[2] = {dL_db2}")
        
        # Propagar erro para camada oculta
        # ∂L/∂a[1] = W[2].T @ ∂L/∂z[2]
        dL_da1 = np.dot(self.W2.T, dL_dz2)
        if verbose:
            print(f"\n--- CAMADA OCULTA ---")
            print(f"∂L/∂a[1] = W[2].T @ ∂L/∂z[2] = {dL_da1}")
        
        # Derivada em relação a z[1]
        # ∂L/∂z[1] = ∂L/∂a[1] * ∂a[1]/∂z[1]
        # onde ∂a[1]/∂z[1] = ReLU'(z[1])
        da1_dz1 = self.relu_derivative(self.z1)
        dL_dz1 = dL_da1 * da1_dz1
        if verbose:
            print(f"\n∂a[1]/∂z[1] = ReLU'(z[1]) = {da1_dz1}")
            print(f"∂L/∂z[1] = ∂L/∂a[1] * ∂a[1]/∂z[1] = {dL_dz1}")
        
        # Gradientes para W[1] e b[1]
        # ∂L/∂W[1] = ∂L/∂z[1] @ x.T
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1
        if verbose:
            print(f"\n∂L/∂W[1] = ∂L/∂z[1] @ x.T = {dL_dW1}")
            print(f"∂L/∂b[1] = ∂L/∂z[1] = {dL_db1}")
        
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    def update_weights(self, dL_dW1, dL_db1, dL_dW2, dL_db2, verbose=True):
        """
        Atualiza os pesos usando Gradient Descent
        W_novo = W_antigo - learning_rate * ∂L/∂W
        """
        if verbose:
            print("\n" + "="*60)
            print("ATUALIZAÇÃO DOS PESOS (Gradient Descent)")
            print("="*60)
            print(f"\nTaxa de aprendizado: {self.learning_rate}")
            print(f"\n--- PESOS ANTES DA ATUALIZAÇÃO ---")
            print(f"W[1]:\n{self.W1}")
            print(f"b[1]: {self.b1}")
            print(f"W[2]: {self.W2}")
            print(f"b[2]: {self.b2}")
        
        # Atualizar pesos
        self.W1 = self.W1 - self.learning_rate * dL_dW1
        self.b1 = self.b1 - self.learning_rate * dL_db1
        self.W2 = self.W2 - self.learning_rate * dL_dW2
        self.b2 = self.b2 - self.learning_rate * dL_db2
        
        if verbose:
            print(f"\n--- PESOS DEPOIS DA ATUALIZAÇÃO ---")
            print(f"W[1]:\n{self.W1}")
            print(f"b[1]: {self.b1}")
            print(f"W[2]: {self.W2}")
            print(f"b[2]: {self.b2}")
    
    def train_step(self, x, y_true, verbose=True):
        """
        Uma iteração completa de treinamento:
        1. Forward pass
        2. Calcular loss
        3. Backward pass
        4. Atualizar pesos
        """
        # Forward pass
        y_pred = self.forward_pass(x, verbose)
        
        # Calcular loss
        loss = self.binary_cross_entropy(y_pred, y_true)
        if verbose:
            print(f"\n--- LOSS ---")
            print(f"Loss (Binary Cross-Entropy) = {loss:.6f}")
        
        # Backward pass
        gradients = self.backward_pass(x, y_true, verbose)
        
        # Atualizar pesos
        self.update_weights(*gradients, verbose)
        
        return loss


def main():
    """Exemplo de uso do MLP com cálculo manual"""
    
    print("\n" + "="*60)
    print("MLP DE 2 CAMADAS - EXEMPLO MANUAL")
    print("="*60)
    print("\nArquitetura:")
    print("- Entrada: 3 neurônios")
    print("- Camada oculta: 2 neurônios (ReLU)")
    print("- Saída: 1 neurônio (Sigmoid)")
    print("- Loss: Binary Cross-Entropy")
    
    # Criar a rede
    mlp = MLPManual()
    
    # Dados de exemplo
    x = np.array([0.5, 0.6, 0.1])  # entrada
    y = np.array([1.0])              # label verdadeiro (classe positiva)
    
    print(f"\n\nDados de treinamento:")
    print(f"Input x: {x}")
    print(f"Label y: {y}")
    
    # Executar uma iteração de treinamento
    loss = mlp.train_step(x, y, verbose=True)
    
    print("\n" + "="*60)
    print("RESUMO DA ITERAÇÃO")
    print("="*60)
    print(f"Loss final: {loss:.6f}")
    print("\nTreinamento de uma iteração completo!")
    print("Os pesos foram atualizados usando backpropagation.")


if __name__ == "__main__":
    main()