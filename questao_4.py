"""
Questão 4: Implementação de Word Embeddings
Explicação e implementação de embeddings iniciais para palavras
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class WordEmbeddings:
    """
    Classe para criar e manipular word embeddings
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Inicializa a camada de embeddings
        
        Args:
            vocab_size: Tamanho do vocabulário
            embedding_dim: Dimensão dos vetores de embedding
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        
    def build_vocab(self, words):
        """
        Constrói o vocabulário a partir de uma lista de palavras
        
        Args:
            words: Lista de palavras únicas
        """
        for idx, word in enumerate(words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def initialize_embeddings(self, method='random'):
        """
        Inicializa os embeddings usando diferentes métodos
        
        Args:
            method: 'random', 'xavier', ou 'normal'
        """
        if method == 'random':
            # Inicialização aleatória uniforme entre -1 e 1
            self.embeddings = np.random.uniform(-1, 1, 
                                               (self.vocab_size, self.embedding_dim))
        
        elif method == 'xavier':
            # Inicialização Xavier/Glorot
            limit = np.sqrt(6 / (self.vocab_size + self.embedding_dim))
            self.embeddings = np.random.uniform(-limit, limit, 
                                               (self.vocab_size, self.embedding_dim))
        
        elif method == 'normal':
            # Inicialização normal com média 0 e desvio padrão pequeno
            self.embeddings = np.random.normal(0, 0.1, 
                                              (self.vocab_size, self.embedding_dim))
        
        print(f"✓ Embeddings inicializados usando método '{method}'")
        print(f"  Shape: {self.embeddings.shape}")
    
    def get_embedding(self, word):
        """Retorna o vetor de embedding para uma palavra"""
        if word not in self.word2idx:
            raise ValueError(f"Palavra '{word}' não está no vocabulário")
        idx = self.word2idx[word]
        return self.embeddings[idx]
    
    def get_embedding_by_idx(self, idx):
        """Retorna o vetor de embedding para um índice"""
        return self.embeddings[idx]
    
    def similarity(self, word1, word2):
        """Calcula a similaridade de cosseno entre duas palavras"""
        emb1 = self.get_embedding(word1).reshape(1, -1)
        emb2 = self.get_embedding(word2).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]
    
    def most_similar(self, word, top_k=5):
        """Encontra as palavras mais similares a uma palavra dada"""
        word_emb = self.get_embedding(word).reshape(1, -1)
        similarities = cosine_similarity(word_emb, self.embeddings)[0]
        
        # Ordenar por similaridade (excluindo a própria palavra)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            word_similar = self.idx2word[idx]
            sim_score = similarities[idx]
            results.append((word_similar, sim_score))
        
        return results
    
    def visualize_embeddings_2d(self, words_to_plot=None):
        """
        Visualiza os embeddings em 2D usando PCA
        
        Args:
            words_to_plot: Lista de palavras para plotar (None = todas)
        """
        if words_to_plot is None:
            words_to_plot = list(self.word2idx.keys())
        
        # Obter embeddings das palavras
        indices = [self.word2idx[w] for w in words_to_plot]
        embeddings_subset = self.embeddings[indices]
        
        # Reduzir para 2D usando PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_subset)
        
        # Plotar
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6)
        
        for i, word in enumerate(words_to_plot):
            plt.annotate(word, 
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=12,
                        ha='center')
        
        plt.title('Visualização dos Word Embeddings (PCA 2D)', fontsize=14)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()


class ManualFeatures:
    """
    Classe para demonstrar features construídas manualmente (one-hot encoding)
    """
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
    
    def build_vocab(self, words):
        """Constrói o vocabulário"""
        for idx, word in enumerate(words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def one_hot_encode(self, word):
        """
        Retorna representação one-hot de uma palavra
        Esta é uma feature manual tradicional
        """
        if word not in self.word2idx:
            raise ValueError(f"Palavra '{word}' não está no vocabulário")
        
        one_hot = np.zeros(self.vocab_size)
        one_hot[self.word2idx[word]] = 1
        return one_hot


def compare_embeddings_vs_manual():
    """
    Demonstra as diferenças entre embeddings e features manuais
    """
    print("\n" + "="*70)
    print("COMPARAÇÃO: Embeddings vs Features Manuais (One-Hot)")
    print("="*70)
    
    # Vocabulário pequeno
    vocab = ['gato', 'cachorro', 'animal', 'carro', 'bicicleta', 
             'veículo', 'feliz', 'triste', 'emoção']
    vocab_size = len(vocab)
    embedding_dim = 5
    
    # 1. Features Manuais (One-Hot)
    print("\n1. FEATURES MANUAIS (One-Hot Encoding):")
    print("-" * 70)
    manual = ManualFeatures(vocab_size)
    manual.build_vocab(vocab)
    
    print(f"   Dimensão: {vocab_size} (igual ao tamanho do vocabulário)")
    print(f"   Representação de 'gato':")
    gato_onehot = manual.one_hot_encode('gato')
    print(f"   {gato_onehot}")
    print(f"\n   Problemas do One-Hot:")
    print(f"   ✗ Alta dimensionalidade (cresce com vocabulário)")
    print(f"   ✗ Vetores esparsos (apenas 1 valor não-zero)")
    print(f"   ✗ Sem relação semântica (todas palavras são ortogonais)")
    
    # Similaridade entre palavras similares e diferentes
    cachorro_onehot = manual.one_hot_encode('cachorro')
    carro_onehot = manual.one_hot_encode('carro')
    
    sim_gato_cachorro = cosine_similarity([gato_onehot], [cachorro_onehot])[0][0]
    sim_gato_carro = cosine_similarity([gato_onehot], [carro_onehot])[0][0]
    
    print(f"\n   Similaridade 'gato' vs 'cachorro': {sim_gato_cachorro:.4f}")
    print(f"   Similaridade 'gato' vs 'carro': {sim_gato_carro:.4f}")
    print(f"   ✗ Todas as palavras têm similaridade 0 (não captura semântica!)")
    
    # 2. Embeddings
    print("\n\n2. WORD EMBEDDINGS:")
    print("-" * 70)
    embeddings = WordEmbeddings(vocab_size, embedding_dim)
    embeddings.build_vocab(vocab)
    embeddings.initialize_embeddings(method='xavier')
    
    print(f"   Dimensão: {embedding_dim} (fixo, independente do vocabulário)")
    print(f"   Representação de 'gato':")
    gato_emb = embeddings.get_embedding('gato')
    print(f"   {gato_emb}")
    print(f"\n   Vantagens dos Embeddings:")
    print(f"   ✓ Dimensionalidade fixa e reduzida")
    print(f"   ✓ Vetores densos (todos valores significativos)")
    print(f"   ✓ PODEM aprender relações semânticas (após treinamento)")
    print(f"   ✓ Melhor generalização")
    
    # Simular embeddings após "treinamento" (ajustando manualmente para demonstração)
    embeddings.embeddings[embeddings.word2idx['gato']] = np.array([0.8, 0.7, 0.1, 0.2, 0.3])
    embeddings.embeddings[embeddings.word2idx['cachorro']] = np.array([0.7, 0.8, 0.15, 0.25, 0.2])
    embeddings.embeddings[embeddings.word2idx['carro']] = np.array([0.1, 0.2, 0.9, 0.8, 0.1])
    
    sim_emb_gato_cachorro = embeddings.similarity('gato', 'cachorro')
    sim_emb_gato_carro = embeddings.similarity('gato', 'carro')
    
    print(f"\n   Após treinamento (simulado):")
    print(f"   Similaridade 'gato' vs 'cachorro': {sim_emb_gato_cachorro:.4f}")
    print(f"   Similaridade 'gato' vs 'carro': {sim_emb_gato_carro:.4f}")
    print(f"   ✓ Captura relações semânticas! (animais são mais similares)")


def exemplo_pratico():
    """
    Exemplo prático completo de uso de embeddings
    """
    print("\n\n" + "="*70)
    print("EXEMPLO PRÁTICO: Sistema de Embeddings")
    print("="*70)
    
    # Vocabulário maior e mais realista
    vocab = [
        'python', 'java', 'javascript', 'linguagem', 'programação',
        'cachorro', 'gato', 'pássaro', 'animal', 'pet',
        'carro', 'bicicleta', 'ônibus', 'transporte', 'veículo',
        'feliz', 'alegre', 'triste', 'emoção', 'sentimento'
    ]
    
    vocab_size = len(vocab)
    embedding_dim = 8
    
    # Criar e inicializar embeddings
    print(f"\n1. Criando sistema de embeddings")
    print(f"   Vocabulário: {vocab_size} palavras")
    print(f"   Dimensão: {embedding_dim}")
    
    we = WordEmbeddings(vocab_size, embedding_dim)
    we.build_vocab(vocab)
    we.initialize_embeddings(method='xavier')
    
    # Mostrar alguns embeddings
    print(f"\n2. Exemplos de embeddings iniciais:")
    for word in ['python', 'cachorro', 'feliz']:
        emb = we.get_embedding(word)
        print(f"   {word}: {emb[:4]}... (primeiros 4 valores)")
    
    # Propriedades dos embeddings
    print(f"\n3. Propriedades dos Embeddings:")
    print(f"   ✓ Cada palavra é representada por {embedding_dim} números reais")
    print(f"   ✓ Embeddings são aprendidos durante o treinamento")
    print(f"   ✓ Palavras similares tendem a ter embeddings similares")
    print(f"   ✓ Permitem operações algébricas (ex: rei - homem + mulher ≈ rainha)")
    
    # Visualizar
    print(f"\n4. Gerando visualização 2D dos embeddings...")
    we.visualize_embeddings_2d()
    print(f"   ✓ Visualização salva em 'embeddings_visualization.png'")
    
    return we


if __name__ == "__main__":
    print("="*70)
    print("QUESTÃO 4: Word Embeddings - Implementação e Explicação")
    print("="*70)
    
    # Parte 1: Comparação
    compare_embeddings_vs_manual()
    
    # Parte 2: Exemplo prático
    we = exemplo_pratico()
    
    print("\n\n" + "="*70)
    print("RESUMO - DIFERENÇAS PRINCIPAIS")
    print("="*70)
    print("""
    FEATURES MANUAIS (One-Hot):
    • Dimensão = tamanho do vocabulário
    • Vetores esparsos (um único 1, resto zeros)
    • Sem informação semântica
    • Não generalizam bem
    • Ortogonais (similaridade sempre 0)
    
    EMBEDDINGS:
    • Dimensão fixa (geralmente 50-300)
    • Vetores densos (todos valores significativos)
    • Capturam relações semânticas
    • Generalizam bem
    • Aprendidos automaticamente dos dados
    • Palavras similares têm vetores similares
    """)
    
    print("\n✓ Implementação completa!")
    print("✓ Execute este código para ver os resultados e visualizações\n")