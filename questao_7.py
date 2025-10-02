import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class SoftmaxLayer:
    """
    Implementação de uma camada Softmax para classificação multiclasse.
    
    A função softmax converte um vetor de valores reais (logits) em 
    probabilidades que somam 1.
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, logits: np.ndarray) -> np.ndarray:
        """
        Aplica a função softmax aos logits de entrada.
        
        Fórmula: softmax(z_i) = exp(z_i) / sum(exp(z_j)) para todo j
        
        Args:
            logits: Array de valores reais (scores não normalizados)
            
        Returns:
            Array de probabilidades que somam 1
        """
        # Subtrai o máximo para estabilidade numérica (evita overflow)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente da camada softmax (para treinamento).
        
        Args:
            grad_output: Gradiente da função de perda
            
        Returns:
            Gradiente em relação aos logits de entrada
        """
        # Jacobiana da softmax
        batch_size = self.output.shape[0]
        grad_input = np.zeros_like(self.output)
        
        for i in range(batch_size):
            s = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])
        
        return grad_input


class SimpleSentimentClassifier:
    """
    Classificador simples de sentimentos usando Softmax.
    
    Classes: 0=Negativo, 1=Neutro, 2=Positivo
    """
    
    def __init__(self, vocab_size: int = 100, embedding_dim: int = 10):
        """
        Inicializa o classificador com pesos aleatórios.
        
        Args:
            vocab_size: Tamanho do vocabulário
            embedding_dim: Dimensão dos embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_classes = 3
        
        # Embeddings de palavras (simplificado)
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Camada de classificação (embedding_dim -> 3 classes)
        self.W = np.random.randn(embedding_dim, self.n_classes) * 0.1
        self.b = np.zeros(self.n_classes)
        
        # Camada softmax
        self.softmax = SoftmaxLayer()
        
        # Vocabulário simples para demonstração
        self.word_to_idx = {
            'ótimo': 0, 'excelente': 1, 'maravilhoso': 2, 'amor': 3, 'adoro': 4,
            'péssimo': 5, 'horrível': 6, 'ruim': 7, 'odeio': 8, 'terrível': 9,
            'ok': 10, 'normal': 11, 'comum': 12, 'médio': 13, 'aceitável': 14,
            'bom': 15, 'legal': 16, 'gostei': 17, 'feliz': 18, 'triste': 19
        }
        self.class_names = ['Negativo', 'Neutro', 'Positivo']
    
    def preprocess(self, sentence: str) -> np.ndarray:
        """
        Converte uma frase em representação vetorial (média dos embeddings).
        
        Args:
            sentence: Frase de entrada
            
        Returns:
            Vetor de características (embedding médio)
        """
        words = sentence.lower().split()
        embeddings_list = []
        
        for word in words:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                embeddings_list.append(self.embeddings[idx])
        
        if embeddings_list:
            # Média dos embeddings das palavras
            return np.mean(embeddings_list, axis=0)
        else:
            # Retorna vetor zero se nenhuma palavra conhecida
            return np.zeros(self.embedding_dim)
    
    def predict(self, sentence: str) -> Tuple[np.ndarray, int, str]:
        """
        Classifica uma frase e retorna probabilidades e previsão.
        
        Args:
            sentence: Frase de entrada
            
        Returns:
            Tupla (probabilidades, classe_predita, nome_classe)
        """
        # Pré-processa a frase
        features = self.preprocess(sentence)
        
        # Calcula logits: z = Wx + b
        logits = np.dot(features, self.W) + self.b
        
        # Aplica softmax para obter probabilidades
        probabilities = self.softmax.forward(logits.reshape(1, -1))[0]
        
        # Classe com maior probabilidade
        predicted_class = np.argmax(probabilities)
        predicted_name = self.class_names[predicted_class]
        
        return probabilities, predicted_class, predicted_name
    
    def visualize_predictions(self, sentence: str):
        """
        Visualiza as probabilidades de classificação.
        
        Args:
            sentence: Frase a ser classificada
        """
        probs, pred_class, pred_name = self.predict(sentence)
        
        plt.figure(figsize=(10, 5))
        
        # Gráfico de barras
        colors = ['#ff6b6b' if i == 0 else '#4ecdc4' if i == 1 else '#95e1d3' 
                  for i in range(3)]
        bars = plt.bar(self.class_names, probs, color=colors, alpha=0.7, edgecolor='black')
        
        # Destaca a classe predita
        bars[pred_class].set_edgecolor('darkgreen')
        bars[pred_class].set_linewidth(3)
        
        plt.ylabel('Probabilidade', fontsize=12)
        plt.title(f'Classificação: "{sentence}"\nPredição: {pred_name}', 
                  fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        
        # Adiciona valores nas barras
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


def demonstrar_softmax():
    """
    Demonstra o funcionamento da função softmax.
    """
    print("=" * 70)
    print("DEMONSTRAÇÃO DA FUNÇÃO SOFTMAX")
    print("=" * 70)
    
    softmax = SoftmaxLayer()
    
    # Exemplo 1: Logits com valores próximos (incerteza)
    logits1 = np.array([[2.0, 2.1, 1.9]])
    probs1 = softmax.forward(logits1)
    
    print("\n1. Logits com valores PRÓXIMOS (modelo incerto):")
    print(f"   Logits:        {logits1[0]}")
    print(f"   Probabilidades: {probs1[0]}")
    print(f"   Soma:          {np.sum(probs1[0]):.6f}")
    print(f"   → Distribuição uniforme indica INCERTEZA")
    
    # Exemplo 2: Logits com valores distintos (confiança)
    logits2 = np.array([[5.0, 1.0, 0.5]])
    probs2 = softmax.forward(logits2)
    
    print("\n2. Logits com valores DISTINTOS (modelo confiante):")
    print(f"   Logits:        {logits2[0]}")
    print(f"   Probabilidades: {probs2[0]}")
    print(f"   Soma:          {np.sum(probs2[0]):.6f}")
    print(f"   → Classe 0 tem alta probabilidade indica CONFIANÇA")
    
    # Exemplo 3: Logits negativos
    logits3 = np.array([[-1.0, -2.0, -0.5]])
    probs3 = softmax.forward(logits3)
    
    print("\n3. Logits NEGATIVOS (também funcionam!):")
    print(f"   Logits:        {logits3[0]}")
    print(f"   Probabilidades: {probs3[0]}")
    print(f"   Soma:          {np.sum(probs3[0]):.6f}")
    
    print("\n" + "=" * 70)
    print("PROPRIEDADES DO SOFTMAX:")
    print("=" * 70)
    print("✓ As probabilidades sempre somam 1.0")
    print("✓ Todas as probabilidades estão entre 0 e 1")
    print("✓ A classe com maior logit tem maior probabilidade")
    print("✓ A diferença entre logits afeta a confiança da predição")
    print("=" * 70)


def exemplo_classificacao():
    """
    Exemplo de uso do classificador de sentimentos.
    """
    print("\n" + "=" * 70)
    print("CLASSIFICADOR DE SENTIMENTOS COM SOFTMAX")
    print("=" * 70)
    
    # Inicializa o classificador
    classifier = SimpleSentimentClassifier()
    
    # Ajusta pesos manualmente para demonstração
    # (em um modelo real, esses pesos seriam aprendidos)
    classifier.W = np.array([
        [-2.5, -0.5, 3.0],   # Dimensão 0: positivo
        [-2.0, -0.3, 2.5],   # Dimensão 1: positivo
        [3.0, 0.2, -2.5],    # Dimensão 2: negativo
        [-1.5, -0.2, 2.0],   # Dimensão 3: positivo
        [2.5, 0.1, -2.0],    # Dimensão 4: negativo
        [0.0, 2.0, 0.0],     # Dimensão 5: neutro
        [-0.5, 1.5, 0.5],    # Dimensão 6: neutro
        [1.0, -0.5, 1.5],    # Dimensão 7: positivo/neutro
        [-1.0, 1.0, 0.5],    # Dimensão 8: neutro
        [0.5, 0.5, -1.0]     # Dimensão 9: negativo
    ])
    
    # Frases de teste
    test_sentences = [
        "ótimo excelente maravilhoso",
        "péssimo horrível terrível",
        "normal ok aceitável",
        "bom legal gostei",
        "ruim triste"
    ]
    
    print("\nClassificando frases...\n")
    
    for sentence in test_sentences:
        probs, pred_class, pred_name = classifier.predict(sentence)
        
        print(f'Frase: "{sentence}"')
        print(f"  Predição: {pred_name}")
        print(f"  Probabilidades:")
        for i, (class_name, prob) in enumerate(zip(classifier.class_names, probs)):
            bar = "█" * int(prob * 50)
            print(f"    {class_name:9s}: {prob:.4f} {bar}")
        print()
    
    print("=" * 70)
    print("COMO INTERPRETAR AS PROBABILIDADES:")
    print("=" * 70)
    print("""
1. MAGNITUDE das probabilidades:
   - Prob > 0.7: Modelo está CONFIANTE na predição
   - Prob ≈ 0.5: Modelo está INCERTO (considere contexto adicional)
   - Prob ≈ 0.33: Modelo NÃO SABE (distribuição uniforme)

2. DIFERENÇA entre probabilidades:
   - Grande diferença: Predição clara e confiável
   - Pequena diferença: Predição ambígua

3. DISTRIBUIÇÃO das probabilidades:
   - Concentrada (ex: [0.9, 0.05, 0.05]): Alta confiança
   - Uniforme (ex: [0.33, 0.34, 0.33]): Baixa confiança

4. USO PRÁTICO:
   - Use a classe com maior probabilidade como predição
   - Considere um threshold (ex: 0.6) para aceitar predições
   - Probabilidades baixas podem indicar outliers ou novos padrões
    """)
    print("=" * 70)


if __name__ == "__main__":
    # Demonstra o funcionamento do softmax
    demonstrar_softmax()
    
    # Exemplo de classificação
    exemplo_classificacao()
    
    print("\n" + "=" * 70)
    print("Para visualizar graficamente, use:")
    print("  classifier = SimpleSentimentClassifier()")
    print('  classifier.visualize_predictions("sua frase aqui")')
    print("=" * 70)