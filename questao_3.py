import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Classe da Rede Neural
class NeuralNetwork:
    def __init__(self, activation_name='sigmoid', input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
        self.activation_name = activation_name
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos com Xavier/He
        if activation_name == 'relu':
            # He initialization para ReLU
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        else:
            # Xavier initialization para sigmoid e tanh
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        # Selecionar funções de ativação
        if activation_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_name == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation_name == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        
        # História do treinamento
        self.loss_history = []
        self.accuracy_history = []
        self.gradient_norm_history = []
        
    def forward(self, X):
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        
        # Camada de saída (sempre sigmoid para classificação binária)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        m = X.shape[0]
        
        # Gradiente da camada de saída
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Gradiente da camada oculta
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Calcular norma dos gradientes
        gradient_norm = np.sqrt(np.sum(dW1**2) + np.sum(dW2**2) + np.sum(db1**2) + np.sum(db2**2))
        self.gradient_norm_history.append(gradient_norm)
        
        # Atualização dos pesos
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calcular loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)
            
            # Calcular acurácia
            pred_class = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_class == y) * 100
            self.accuracy_history.append(accuracy)
            
            # Backward pass
            self.backward(X, y)
            
            # Parar se convergiu
            if accuracy == 100 and loss < 0.01:
                print(f"{self.activation_name}: Convergiu em {epoch} épocas")
                # Preencher o resto com os valores finais
                for _ in range(epochs - epoch - 1):
                    self.loss_history.append(loss)
                    self.accuracy_history.append(accuracy)
                    self.gradient_norm_history.append(self.gradient_norm_history[-1])
                break
        
        return self.loss_history, self.accuracy_history

# Gerar dados XOR
def generate_xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y

# Executar experimento
def run_experiment(epochs=1000, n_runs=5):
    X, y = generate_xor_data()
    
    activations = ['sigmoid', 'tanh', 'relu']
    results = {act: {'losses': [], 'accuracies': [], 'gradients': [], 'convergence_epochs': []} 
               for act in activations}
    
    print("=" * 60)
    print("EXPERIMENTO: Comparação de Funções de Ativação")
    print("=" * 60)
    print(f"Problema: XOR (Classificação Binária)")
    print(f"Arquitetura: 2 → 4 → 1")
    print(f"Épocas máximas: {epochs}")
    print(f"Número de execuções: {n_runs}")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"\nExecução {run + 1}/{n_runs}:")
        
        for activation in activations:
            np.random.seed(42 + run)  # Para reprodutibilidade
            
            nn = NeuralNetwork(activation_name=activation, learning_rate=0.5)
            losses, accuracies = nn.train(X, y, epochs=epochs)
            
            results[activation]['losses'].append(losses)
            results[activation]['accuracies'].append(accuracies)
            results[activation]['gradients'].append(nn.gradient_norm_history)
            
            # Encontrar época de convergência
            conv_epoch = next((i for i, acc in enumerate(accuracies) if acc == 100), epochs)
            results[activation]['convergence_epochs'].append(conv_epoch)
    
    return results

# Visualizar resultados
def plot_results(results, epochs=1000):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {'sigmoid': '#8b5cf6', 'tanh': '#3b82f6', 'relu': '#10b981'}
    activations = ['sigmoid', 'tanh', 'relu']
    
    # Calcular médias
    avg_results = {}
    for act in activations:
        avg_results[act] = {
            'loss': np.mean(results[act]['losses'], axis=0),
            'accuracy': np.mean(results[act]['accuracies'], axis=0),
            'gradient': np.mean(results[act]['gradients'], axis=0),
            'loss_std': np.std(results[act]['losses'], axis=0),
            'accuracy_std': np.std(results[act]['accuracies'], axis=0)
        }
    
    # 1. Loss ao longo das épocas
    ax1 = fig.add_subplot(gs[0, :])
    for act in activations:
        epochs_range = range(len(avg_results[act]['loss']))
        ax1.plot(epochs_range, avg_results[act]['loss'], 
                label=act.upper(), color=colors[act], linewidth=2)
        ax1.fill_between(epochs_range, 
                        avg_results[act]['loss'] - avg_results[act]['loss_std'],
                        avg_results[act]['loss'] + avg_results[act]['loss_std'],
                        alpha=0.2, color=colors[act])
    ax1.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('Perda ao Longo do Treinamento', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Acurácia ao longo das épocas
    ax2 = fig.add_subplot(gs[1, :])
    for act in activations:
        epochs_range = range(len(avg_results[act]['accuracy']))
        ax2.plot(epochs_range, avg_results[act]['accuracy'], 
                label=act.upper(), color=colors[act], linewidth=2)
        ax2.fill_between(epochs_range,
                        avg_results[act]['accuracy'] - avg_results[act]['accuracy_std'],
                        avg_results[act]['accuracy'] + avg_results[act]['accuracy_std'],
                        alpha=0.2, color=colors[act])
    ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Acurácia ao Longo do Treinamento', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # 3. Norma dos gradientes
    ax3 = fig.add_subplot(gs[2, 0])
    for act in activations:
        epochs_range = range(len(avg_results[act]['gradient']))
        ax3.plot(epochs_range, avg_results[act]['gradient'], 
                label=act.upper(), color=colors[act], linewidth=2)
    ax3.set_xlabel('Época', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Norma do Gradiente', fontsize=11, fontweight='bold')
    ax3.set_title('Magnitude dos Gradientes', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Box plot - Épocas para convergência
    ax4 = fig.add_subplot(gs[2, 1])
    conv_data = [results[act]['convergence_epochs'] for act in activations]
    bp = ax4.boxplot(conv_data, labels=[act.upper() for act in activations],
                     patch_artist=True)
    for patch, act in zip(bp['boxes'], activations):
        patch.set_facecolor(colors[act])
        patch.set_alpha(0.7)
    ax4.set_ylabel('Épocas até Convergência', fontsize=11, fontweight='bold')
    ax4.set_title('Velocidade de Convergência', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Tabela de estatísticas
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    stats_text = "ESTATÍSTICAS FINAIS\n" + "="*30 + "\n\n"
    for act in activations:
        mean_conv = np.mean(results[act]['convergence_epochs'])
        std_conv = np.std(results[act]['convergence_epochs'])
        final_loss = np.mean([losses[-1] for losses in results[act]['losses']])
        final_acc = np.mean([accs[-1] for accs in results[act]['accuracies']])
        
        stats_text += f"{act.upper()}:\n"
        stats_text += f"  Convergência: {mean_conv:.1f} ± {std_conv:.1f} épocas\n"
        stats_text += f"  Loss final: {final_loss:.4f}\n"
        stats_text += f"  Acurácia final: {final_acc:.1f}%\n\n"
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Comparação Empírica de Funções de Ativação - Problema XOR', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('activation_comparison_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico salvo como 'activation_comparison_results.png'")
    plt.show()

# Gerar relatório textual
def generate_report(results):
    activations = ['sigmoid', 'tanh', 'relu']
    
    print("\n" + "="*60)
    print("RELATÓRIO FINAL")
    print("="*60)
    
    # Calcular estatísticas
    stats = {}
    for act in activations:
        conv_epochs = results[act]['convergence_epochs']
        final_losses = [losses[-1] for losses in results[act]['losses']]
        final_accs = [accs[-1] for accs in results[act]['accuracies']]
        
        stats[act] = {
            'mean_conv': np.mean(conv_epochs),
            'std_conv': np.std(conv_epochs),
            'min_conv': np.min(conv_epochs),
            'max_conv': np.max(conv_epochs),
            'mean_loss': np.mean(final_losses),
            'mean_acc': np.mean(final_accs)
        }
    
    # Ordenar por convergência
    sorted_acts = sorted(activations, key=lambda x: stats[x]['mean_conv'])
    
    print("\n1. VELOCIDADE DE CONVERGÊNCIA (épocas até 100% acurácia):")
    print("-" * 60)
    for i, act in enumerate(sorted_acts, 1):
        s = stats[act]
        print(f"{i}º lugar: {act.upper()}")
        print(f"   Média: {s['mean_conv']:.1f} ± {s['std_conv']:.1f} épocas")
        print(f"   Intervalo: [{s['min_conv']}, {s['max_conv']}] épocas\n")
    
    print("\n2. DESEMPENHO FINAL:")
    print("-" * 60)
    for act in activations:
        s = stats[act]
        print(f"{act.upper()}:")
        print(f"   Loss final: {s['mean_loss']:.4f}")
        print(f"   Acurácia final: {s['mean_acc']:.1f}%\n")
    
    print("\n3. CONCLUSÃO:")
    print("-" * 60)
    best = sorted_acts[0]
    print(f"A função de ativação {best.upper()} apresentou a MELHOR convergência,")
    print(f"atingindo 100% de acurácia em média com {stats[best]['mean_conv']:.1f} épocas.")
    
    print("\n4. ANÁLISE COMPARATIVA:")
    print("-" * 60)
    
    if best == 'relu':
        print("• ReLU convergiu mais rapidamente devido a:")
        print("  - Gradientes não saturados para valores positivos")
        print("  - Ausência de vanishing gradient")
        print("  - Derivada constante (1) para x > 0")
    elif best == 'tanh':
        print("• Tanh convergiu mais rapidamente devido a:")
        print("  - Centralização em zero (saídas entre -1 e 1)")
        print("  - Gradientes mais fortes que sigmoid")
        print("  - Melhor fluxo de gradiente que sigmoid")
    else:
        print("• Sigmoid teve desempenho inferior devido a:")
        print("  - Vanishing gradient problem")
        print("  - Saídas não centralizadas em zero")
        print("  - Gradientes muito pequenos nas extremidades")
    
    print("\n" + "="*60)
    
    # Salvar relatório em arquivo
    with open('relatorio_experimento.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DO EXPERIMENTO\n")
        f.write("="*60 + "\n\n")
        f.write(f"Melhor ativação: {best.upper()}\n")
        f.write(f"Convergência média: {stats[best]['mean_conv']:.1f} épocas\n\n")
        
        for act in sorted_acts:
            s = stats[act]
            f.write(f"\n{act.upper()}:\n")
            f.write(f"  Convergência: {s['mean_conv']:.1f} ± {s['std_conv']:.1f} épocas\n")
            f.write(f"  Loss final: {s['mean_loss']:.4f}\n")
            f.write(f"  Acurácia: {s['mean_acc']:.1f}%\n")
    
    print("\n✓ Relatório salvo como 'relatorio_experimento.txt'")

# Função principal
def main():
    # Executar experimento
    results = run_experiment(epochs=1000, n_runs=5)
    
    # Gerar visualizações
    plot_results(results, epochs=1000)
    
    # Gerar relatório
    generate_report(results)
    
    print("\n" + "="*60)
    print("EXPERIMENTO CONCLUÍDO!")
    print("="*60)
    print("\nArquivos gerados:")
    print("  • activation_comparison_results.png")
    print("  • relatorio_experimento.txt")
    print("\n")

if __name__ == "__main__":
    main()