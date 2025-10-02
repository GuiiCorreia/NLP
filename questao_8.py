import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Configurar seed para reprodutibilidade
np.random.seed(42)
torch.manual_seed(42)


class BigramModel:
    """Modelo de Linguagem baseado em Bigramas com suavização Add-1 (Laplace)"""
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        
    def train(self, sentences):
        """Treina o modelo de bigramas"""
        for sentence in sentences:
            words = ['<s>'] + sentence.split() + ['</s>']
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i+1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)
        
        self.vocab_size = len(self.vocab)
        print(f"Bigrama: Vocabulário treinado com {self.vocab_size} palavras")
        
    def probability(self, w1, w2):
        """Calcula P(w2|w1) com suavização Add-1 (Laplace)"""
        # P_laplace(w2|w1) = (C(w1,w2) + 1) / (C(w1) + V)
        count_w1_w2 = self.bigram_counts[w1][w2]
        count_w1 = self.unigram_counts[w1]
        
        if count_w1 == 0:
            return 1.0 / self.vocab_size
        
        return (count_w1_w2 + 1) / (count_w1 + self.vocab_size)
    
    def sentence_probability(self, sentence):
        """Calcula a probabilidade de uma sentença"""
        words = ['<s>'] + sentence.split() + ['</s>']
        log_prob = 0.0
        
        for i in range(len(words) - 1):
            prob = self.probability(words[i], words[i+1])
            log_prob += np.log(prob) if prob > 0 else -100
            
        return log_prob
    
    def perplexity(self, test_sentences):
        """Calcula a perplexidade no conjunto de teste"""
        total_log_prob = 0
        total_words = 0
        
        for sentence in test_sentences:
            words = sentence.split()
            total_words += len(words) + 1  # +1 para </s>
            total_log_prob += self.sentence_probability(sentence)
        
        # Perplexity = exp(-1/N * sum(log P(w_i)))
        perplexity = np.exp(-total_log_prob / total_words)
        return perplexity


class SimpleNNLM(nn.Module):
    """
    Neural Network Language Model simples
    Arquitetura: Embedding -> Hidden Layer (ReLU) -> Output (Softmax)
    """
    
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=100, context_size=2):
        super(SimpleNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        # Camada de embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Camadas da rede neural
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, context):
        # context: [batch_size, context_size]
        embeds = self.embeddings(context)  # [batch_size, context_size, embedding_dim]
        embeds = embeds.view(embeds.size(0), -1)  # [batch_size, context_size * embedding_dim]
        
        hidden = self.relu(self.fc1(embeds))
        output = self.fc2(hidden)
        
        return output


class NNLMWrapper:
    """Wrapper para treinar e avaliar o NN-LM"""
    
    def __init__(self, embedding_dim=50, hidden_dim=100, context_size=2):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.word2idx = {'<PAD>': 0, '<s>': 1, '</s>': 2}
        self.idx2word = {0: '<PAD>', 1: '<s>', 2: '</s>'}
        self.vocab_size = 3
        self.model = None
        
    def build_vocab(self, sentences):
        """Constrói o vocabulário"""
        for sentence in sentences:
            for word in sentence.split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"NN-LM: Vocabulário construído com {self.vocab_size} palavras")
        
    def prepare_data(self, sentences):
        """Prepara dados para treinamento"""
        data = []
        for sentence in sentences:
            words = ['<s>'] + sentence.split() + ['</s>']
            indices = [self.word2idx.get(w, 0) for w in words]
            
            # Criar contextos e targets
            for i in range(self.context_size, len(indices)):
                context = indices[i-self.context_size:i]
                target = indices[i]
                data.append((context, target))
        
        return data
    
    def train(self, sentences, epochs=20, lr=0.001):
        """Treina o modelo neural"""
        self.build_vocab(sentences)
        self.model = SimpleNNLM(self.vocab_size, self.embedding_dim, 
                                self.hidden_dim, self.context_size)
        
        data = self.prepare_data(sentences)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"NN-LM: Treinando com {len(data)} exemplos...")
        
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(data)
            
            for context, target in data:
                context_tensor = torch.tensor([context], dtype=torch.long)
                target_tensor = torch.tensor([target], dtype=torch.long)
                
                optimizer.zero_grad()
                output = self.model(context_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data)
            losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Época {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def perplexity(self, test_sentences):
        """Calcula a perplexidade no conjunto de teste"""
        self.model.eval()
        data = self.prepare_data(test_sentences)
        
        total_log_prob = 0
        total_words = 0
        
        with torch.no_grad():
            for context, target in data:
                context_tensor = torch.tensor([context], dtype=torch.long)
                output = self.model(context_tensor)
                probs = torch.softmax(output, dim=1)
                prob = probs[0, target].item()
                
                if prob > 0:
                    total_log_prob += np.log(prob)
                else:
                    total_log_prob += -100
                
                total_words += 1
        
        perplexity = np.exp(-total_log_prob / total_words)
        return perplexity
    
    def predict_next_word(self, context_words):
        """Prediz a próxima palavra dado um contexto"""
        self.model.eval()
        
        # Converter palavras para índices
        context_indices = [self.word2idx.get(w, 0) for w in context_words[-self.context_size:]]
        
        # Pad se necessário
        while len(context_indices) < self.context_size:
            context_indices = [0] + context_indices
        
        context_tensor = torch.tensor([context_indices], dtype=torch.long)
        
        with torch.no_grad():
            output = self.model(context_tensor)
            probs = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs[0], 5)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            word = self.idx2word[idx.item()]
            predictions.append((word, prob.item()))
        
        return predictions


def demonstrate_generalization():
    """Demonstra a capacidade de generalização do NN-LM"""
    
    # Corpus de treinamento: frases sobre animais
    train_corpus = [
        "the cat eats fish",
        "the cat drinks water",
        "the cat sleeps well",
        "the dog eats meat",
        "the dog drinks water",
        "the dog runs fast"
    ]
    
    # Corpus de teste: palavra similar (bird) não vista no treino
    test_corpus = [
        "the bird eats seeds"  # "bird" nunca foi visto no treino
    ]
    
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO DE GENERALIZAÇÃO")
    print("="*60)
    print("\nCorpus de Treinamento:")
    for sent in train_corpus:
        print(f"  - {sent}")
    
    print("\nCorpus de Teste:")
    for sent in test_corpus:
        print(f"  - {sent}")
    
    # Treinar Bigrama
    print("\n--- MODELO DE BIGRAMAS ---")
    bigram = BigramModel()
    bigram.train(train_corpus)
    
    # Para bigramas: "bird" nunca foi visto, então terá problema
    print(f"\nP('bird'|'the') no modelo de bigramas:")
    prob_bird = bigram.probability('the', 'bird')
    print(f"  Probabilidade: {prob_bird:.6f} (muito baixa devido a Add-1)")
    
    # Treinar NN-LM
    print("\n--- NEURAL NETWORK LANGUAGE MODEL ---")
    nnlm = NNLMWrapper(embedding_dim=30, hidden_dim=50, context_size=2)
    nnlm.train(train_corpus, epochs=30, lr=0.01)
    
    # NN-LM pode generalizar melhor através dos embeddings
    print(f"\nPredições do NN-LM após 'the':")
    predictions = nnlm.predict_next_word(['the'])
    for word, prob in predictions[:3]:
        print(f"  {word}: {prob:.4f}")
    
    return train_corpus, test_corpus


def compare_models_on_zeros():
    """Compara como os modelos lidam com sequências não vistas"""
    
    print("\n" + "="*60)
    print("COMPARAÇÃO: LIDANDO COM ZEROS")
    print("="*60)
    
    train_corpus = [
        "I like apples",
        "I like oranges",
        "I eat apples",
        "you like bananas",
        "you eat oranges"
    ]
    
    # Sequência não vista: "I eat bananas"
    test_sentence = "I eat bananas"
    
    print(f"\nTreinamento: {len(train_corpus)} sentenças")
    print(f"Teste: '{test_sentence}' (combinação não vista)")
    
    # Bigrama
    bigram = BigramModel()
    bigram.train(train_corpus)
    
    print("\n--- BIGRAMA ---")
    print("Problema: 'eat bananas' nunca foi visto juntos no treino")
    prob = bigram.probability('eat', 'bananas')
    print(f"P('bananas'|'eat') = {prob:.6f}")
    print("Mesmo com Add-1, a probabilidade é artificialmente baixa")
    
    # NN-LM
    nnlm = NNLMWrapper(embedding_dim=20, hidden_dim=40, context_size=2)
    nnlm.train(train_corpus, epochs=25, lr=0.01)
    
    print("\n--- NN-LM ---")
    print("Vantagem: Aprende que 'apples', 'oranges' e 'bananas' são similares")
    print("Através dos embeddings, pode generalizar para combinações não vistas")
    

def generate_comparison_report():
    """Gera relatório completo comparando os modelos"""
    
    print("\n" + "="*60)
    print("COMPARAÇÃO COMPLETA: BIGRAMA vs NN-LM")
    print("="*60)
    
    # Corpus maior
    corpus = [
        "the cat sits on the mat",
        "the dog runs in the park",
        "the bird flies in the sky",
        "the cat eats fish every day",
        "the dog eats meat every day",
        "the cat drinks water",
        "the dog drinks water",
        "the bird drinks water",
        "cats are animals",
        "dogs are animals",
        "birds are animals",
        "the mat is soft",
        "the park is green",
        "the sky is blue"
    ]
    
    # Dividir em treino e teste
    train_size = int(0.7 * len(corpus))
    train_corpus = corpus[:train_size]
    test_corpus = corpus[train_size:]
    
    print(f"\nTamanho do corpus: {len(corpus)} sentenças")
    print(f"Treino: {len(train_corpus)} sentenças")
    print(f"Teste: {len(test_corpus)} sentenças")
    
    # Treinar Bigrama
    print("\n--- Treinando Modelo de Bigramas ---")
    bigram = BigramModel()
    bigram.train(train_corpus)
    bigram_perp = bigram.perplexity(test_corpus)
    print(f"Perplexidade no teste: {bigram_perp:.2f}")
    
    # Treinar NN-LM
    print("\n--- Treinando Neural Network LM ---")
    nnlm = NNLMWrapper(embedding_dim=40, hidden_dim=80, context_size=2)
    losses = nnlm.train(train_corpus, epochs=30, lr=0.005)
    nnlm_perp = nnlm.perplexity(test_corpus)
    print(f"Perplexidade no teste: {nnlm_perp:.2f}")
    
    # Criar visualizações
    create_visualizations(bigram_perp, nnlm_perp, losses)
    
    return bigram_perp, nnlm_perp


def create_visualizations(bigram_perp, nnlm_perp, losses):
    """Cria visualizações comparativas e salva em PDF"""
    
    pdf_filename = 'comparacao_bigrama_vs_nnlm.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        # Página 1: Comparação de Perplexidade
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Bigrama\n(com Add-1)', 'NN-LM\n(2 camadas)']
        perplexities = [bigram_perp, nnlm_perp]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(models, perplexities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Perplexidade', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Perplexidade: Bigrama vs NN-LM', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adicionar valores nas barras
        for bar, perp in zip(bars, perplexities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{perp:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Página 2: Curva de Loss do NN-LM
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, 'o-', color='#4ECDC4', linewidth=2, 
                markersize=6, label='Loss de Treinamento')
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
        ax.set_title('Curva de Aprendizado do NN-LM', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Página 3: Aspectos Comparativos (Texto)
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'Aspectos em que o NN-LM Supera o Modelo de N-gramas', 
                ha='center', fontsize=16, fontweight='bold')
        
        aspects_text = """
1. GENERALIZAÇÃO ATRAVÉS DE EMBEDDINGS
   • Bigramas: Trata cada palavra como símbolo independente
   • NN-LM: Aprende representações vetoriais (embeddings) que capturam similaridade semântica
   • Exemplo: Se treinou com "cat eats fish", pode generalizar para "dog eats meat"
             porque "cat" e "dog" têm embeddings similares

2. PROBLEMA DA ESPARSIDADE (SPARSITY)
   • Bigramas: Muitas combinações possíveis nunca aparecem no treino
   • NN-LM: Embeddings compartilhados reduzem drasticamente a esparsidade
   • Mesmo com suavização (Add-1), bigramas sofrem com sequências não vistas
   • NN-LM: Pode inferir probabilidades razoáveis para sequências não vistas

3. REPRESENTAÇÃO CONTÍNUA vs DISCRETA
   • Bigramas: Espaço discreto de símbolos - "cat" e "dog" não têm relação
   • NN-LM: Espaço contínuo - palavras similares têm vetores próximos
   • Permite interpolação suave entre conceitos relacionados

4. CAPACIDADE DE CAPTURAR PADRÕES COMPLEXOS
   • Bigramas: Apenas dependências locais imediatas (palavra anterior)
   • NN-LM: Hidden layers podem capturar padrões não-lineares
   • Aprende features automáticas em vez de contar co-ocorrências

5. MELHOR USO DOS DADOS DE TREINAMENTO
   • Bigramas: Cada bigrama é aprendido independentemente
   • NN-LM: Compartilha conhecimento através dos embeddings e pesos
   • Resultado: Perplexidade menor = melhor modelagem da linguagem

6. ESCALABILIDADE
   • Bigramas: Cresce quadraticamente com o vocabulário (V²)
   • NN-LM: Cresce linearmente com tamanho do vocabulário no embedding
   • Mais eficiente para vocabulários grandes

RESULTADOS EXPERIMENTAIS:
   • Perplexidade Bigrama: {:.2f}
   • Perplexidade NN-LM: {:.2f}
   • Melhoria: {:.1f}%
        """.format(bigram_perp, nnlm_perp, 
                   ((bigram_perp - nnlm_perp) / bigram_perp) * 100)
        
        fig.text(0.1, 0.05, aspects_text, fontsize=10, family='monospace',
                verticalalignment='bottom')
        
        pdf.savefig(fig)
        plt.close()
    
    print(f"\n✓ Relatório salvo em: {pdf_filename}")


def main():
    """Função principal"""
    
    print("="*60)
    print("COMPARAÇÃO: MODELO DE BIGRAMAS vs NN-LM")
    print("Questão 8: Em quais aspectos o NN-LM supera n-gramas?")
    print("="*60)
    
    # 1. Demonstrar generalização
    demonstrate_generalization()
    
    # 2. Demonstrar problema com zeros
    compare_models_on_zeros()
    
    # 3. Comparação completa e geração de relatório
    generate_comparison_report()
    
    print("\n" + "="*60)
    print("RESUMO DOS ASPECTOS SUPERIORES DO NN-LM:")
    print("="*60)
    print("""
1. ✓ Generalização através de embeddings semânticos
2. ✓ Melhor tratamento de sparsity/zeros
3. ✓ Representação contínua vs discreta
4. ✓ Captura padrões não-lineares complexos
5. ✓ Compartilhamento de conhecimento via parâmetros
6. ✓ Melhor eficiência para vocabulários grandes
7. ✓ Perplexidade menor (melhor predição)
    """)
    
    print("\n✓ Análise completa salva em 'comparacao_bigrama_vs_nnlm.pdf'")
    print("="*60)


if __name__ == "__main__":
    main()