import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

class TextDataset(Dataset):
    """Dataset para janelas deslizantes de texto"""
    def __init__(self, text, window_size, vocab):
        self.window_size = window_size
        self.vocab = vocab
        self.data = []
        
        # Tokenizar texto
        tokens = text.lower().split()
        
        # Criar janelas deslizantes
        for i in range(len(tokens) - window_size):
            context = tokens[i:i + window_size]
            target = tokens[i + window_size]
            
            # Converter para índices
            context_ids = [vocab.get(w, vocab['<UNK>']) for w in context]
            target_id = vocab.get(target, vocab['<UNK>'])
            
            self.data.append((context_ids, target_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)


class FeedForwardLM(nn.Module):
    """Modelo de Linguagem Feedforward com janelas deslizantes"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, window_size):
        super(FeedForwardLM, self).__init__()
        self.window_size = window_size
        
        # Camada de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Camadas feedforward
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, window_size)
        batch_size = x.size(0)
        
        # Embedding: (batch_size, window_size, embedding_dim)
        embeds = self.embedding(x)
        
        # Flatten: (batch_size, window_size * embedding_dim)
        embeds = embeds.view(batch_size, -1)
        
        # Feedforward
        hidden = self.fc1(embeds)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        
        return output


def build_vocab(text, min_freq=2):
    """Constrói vocabulário do texto"""
    tokens = text.lower().split()
    counter = Counter(tokens)
    
    # Adicionar tokens especiais
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def calculate_perplexity(model, dataloader, criterion, device):
    """Calcula perplexidade do modelo"""
    model.eval()
    total_loss = 0
    total_words = 0
    
    with torch.no_grad():
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            
            output = model(context)
            loss = criterion(output, target)
            
            total_loss += loss.item() * target.size(0)
            total_words += target.size(0)
    
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    
    return perplexity


def train_model(model, train_loader, val_loader, epochs, lr, device):
    """Treina o modelo"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_perplexities = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for context, target in train_loader:
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        val_perplexity = calculate_perplexity(model, val_loader, criterion, device)
        
        train_losses.append(avg_train_loss)
        val_perplexities.append(val_perplexity)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Perplexity: {val_perplexity:.2f}')
    
    return train_losses, val_perplexities


def main():
    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Texto de exemplo (você pode carregar seu próprio texto)
    sample_text = """
    the cat sat on the mat . the dog sat on the log . the bird flew over the tree .
    a quick brown fox jumps over the lazy dog . the sun shines bright in the sky .
    machine learning is a subset of artificial intelligence . deep learning uses neural networks .
    natural language processing helps computers understand human language .
    python is a popular programming language for data science .
    the model learns patterns from the training data .
    """ * 50  # Repetir para ter mais dados
    
    # Construir vocabulário
    vocab = build_vocab(sample_text, min_freq=2)
    vocab_size = len(vocab)
    print(f"Tamanho do vocabulário: {vocab_size}")
    
    # Testar diferentes tamanhos de janela
    window_sizes = [2, 3, 4, 5, 6]
    results = {}
    
    # Hiperparâmetros
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 32
    epochs = 30
    lr = 0.001
    
    for window_size in window_sizes:
        print(f"\n{'='*50}")
        print(f"Treinando com janela de tamanho {window_size}")
        print(f"{'='*50}")
        
        # Criar datasets
        dataset = TextDataset(sample_text, window_size, vocab)
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Criar modelo
        model = FeedForwardLM(vocab_size, embedding_dim, hidden_dim, window_size).to(device)
        
        # Treinar
        train_losses, val_perplexities = train_model(
            model, train_loader, val_loader, epochs, lr, device
        )
        
        # Salvar resultados
        results[window_size] = {
            'train_losses': train_losses,
            'val_perplexities': val_perplexities,
            'final_perplexity': val_perplexities[-1]
        }
        
        print(f"Perplexidade final: {val_perplexities[-1]:.2f}")
    
    # Plotar resultados
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Perplexidade por época
    plt.subplot(1, 3, 1)
    for window_size in window_sizes:
        plt.plot(results[window_size]['val_perplexities'], 
                label=f'Janela {window_size}', marker='o', markersize=3)
    plt.xlabel('Época')
    plt.ylabel('Perplexidade')
    plt.title('Perplexidade ao Longo do Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Perplexidade final vs tamanho da janela
    plt.subplot(1, 3, 2)
    final_perplexities = [results[ws]['final_perplexity'] for ws in window_sizes]
    plt.bar(window_sizes, final_perplexities, color='steelblue', alpha=0.7)
    plt.xlabel('Tamanho da Janela')
    plt.ylabel('Perplexidade Final')
    plt.title('Perplexidade Final vs Tamanho da Janela')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for i, (ws, pp) in enumerate(zip(window_sizes, final_perplexities)):
        plt.text(ws, pp, f'{pp:.1f}', ha='center', va='bottom')
    
    # Subplot 3: Loss de treinamento
    plt.subplot(1, 3, 3)
    for window_size in window_sizes:
        plt.plot(results[window_size]['train_losses'], 
                label=f'Janela {window_size}', alpha=0.7)
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss de Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('perplexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Resumo dos resultados
    print(f"\n{'='*60}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*60}")
    print(f"{'Tamanho Janela':<20} {'Perplexidade Final':<20} {'Melhoria (%)':<20}")
    print(f"{'-'*60}")
    
    baseline = results[window_sizes[0]]['final_perplexity']
    for ws in window_sizes:
        pp = results[ws]['final_perplexity']
        improvement = ((baseline - pp) / baseline) * 100
        print(f"{ws:<20} {pp:<20.2f} {improvement:<20.2f}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()