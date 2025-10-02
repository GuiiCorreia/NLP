import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mean_pooling(embeddings):
    """
    Implementa mean pooling: calcula a média de cada dimensão ao longo da sequência.
    
    Args:
        embeddings: numpy array de forma (sequence_length, embedding_dim)
    
    Returns:
        numpy array de forma (embedding_dim,) representando a média
    """
    return np.mean(embeddings, axis=0)

def max_pooling(embeddings):
    """
    Implementa max pooling: pega o valor máximo de cada dimensão ao longo da sequência.
    
    Args:
        embeddings: numpy array de forma (sequence_length, embedding_dim)
    
    Returns:
        numpy array de forma (embedding_dim,) representando o máximo
    """
    return np.max(embeddings, axis=0)

def calcular_preservacao_informacao(embeddings, pooled_vector, metodo):
    """
    Calcula métricas de preservação de informação.
    
    Args:
        embeddings: matriz original de embeddings
        pooled_vector: vetor resultante do pooling
        metodo: string indicando o método usado
    
    Returns:
        dict com várias métricas
    """
    # Variância preservada (comparando variância original com variância do resultado)
    variancia_original = np.var(embeddings)
    variancia_pooled = np.var(pooled_vector)
    
    # Magnitude do vetor
    norma_pooled = np.linalg.norm(pooled_vector)
    
    # Informação sobre valores extremos
    valores_max_originais = np.max(embeddings, axis=0)
    valores_min_originais = np.min(embeddings, axis=0)
    
    return {
        'metodo': metodo,
        'variancia_original': variancia_original,
        'variancia_pooled': variancia_pooled,
        'norma': norma_pooled,
        'valores_max': np.max(pooled_vector),
        'valores_min': np.min(pooled_vector),
        'range': np.max(pooled_vector) - np.min(pooled_vector)
    }

def visualizar_comparacao(embeddings, mean_result, max_result):
    """
    Cria visualizações para comparar os métodos de pooling.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Heatmap da matriz original
    sns.heatmap(embeddings.T, cmap='coolwarm', center=0, ax=axes[0, 0], 
                cbar_kws={'label': 'Valor'})
    axes[0, 0].set_title('Matriz de Embeddings Original\n(cada coluna = token, cada linha = dimensão)')
    axes[0, 0].set_xlabel('Tokens na Sequência')
    axes[0, 0].set_ylabel('Dimensões do Embedding')
    
    # 2. Comparação dos vetores resultantes
    x_pos = np.arange(len(mean_result))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, mean_result, width, label='Mean Pooling', alpha=0.8)
    axes[0, 1].bar(x_pos + width/2, max_result, width, label='Max Pooling', alpha=0.8)
    axes[0, 1].set_title('Comparação: Mean vs Max Pooling')
    axes[0, 1].set_xlabel('Dimensão do Embedding')
    axes[0, 1].set_ylabel('Valor')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribuição dos valores originais vs pooled
    axes[1, 0].hist(embeddings.flatten(), bins=30, alpha=0.5, label='Original', density=True)
    axes[1, 0].hist(mean_result, bins=15, alpha=0.5, label='Mean Pooling', density=True)
    axes[1, 0].hist(max_result, bins=15, alpha=0.5, label='Max Pooling', density=True)
    axes[1, 0].set_title('Distribuição dos Valores')
    axes[1, 0].set_xlabel('Valor')
    axes[1, 0].set_ylabel('Densidade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Análise por dimensão
    for i in range(embeddings.shape[1]):
        axes[1, 1].plot(embeddings[:, i], alpha=0.3, color='gray')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1, 1].plot(mean_result, 'o-', label='Mean Pooling', linewidth=2, markersize=8)
    axes[1, 1].plot(max_result, 's-', label='Max Pooling', linewidth=2, markersize=8)
    axes[1, 1].set_title('Valores por Dimensão\n(linhas cinzas = valores originais de cada token)')
    axes[1, 1].set_xlabel('Dimensão do Embedding')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pooling_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualização salva como 'pooling_comparison.png'")
    plt.show()

def main():
    # Exemplo: criar uma matriz de embeddings simulada
    # Representa uma frase de 5 tokens com embeddings de dimensão 8
    np.random.seed(42)
    
    # Simulando embeddings com diferentes características
    # Token 1: valores positivos moderados
    # Token 2: valores negativos
    # Token 3: valores mistos com alguns picos
    # Token 4: valores próximos de zero
    # Token 5: valores positivos altos
    
    embeddings = np.array([
        [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.3],  # Token 1
        [-0.3, -0.5, -0.2, -0.4, -0.1, -0.6, -0.3, -0.2],  # Token 2
        [0.9, -0.4, 0.7, -0.8, 0.5, 0.3, -0.6, 0.8],  # Token 3 (picos)
        [0.1, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1, 0.0],  # Token 4 (quase zero)
        [0.8, 0.9, 0.7, 0.85, 0.75, 0.8, 0.9, 0.85],  # Token 5 (altos)
    ])
    
    print("="*70)
    print("ANÁLISE DE POOLING STRATEGIES PARA EMBEDDINGS")
    print("="*70)
    print(f"\nMatriz de embeddings (shape: {embeddings.shape})")
    print(f"- {embeddings.shape[0]} tokens na sequência")
    print(f"- {embeddings.shape[1]} dimensões por embedding\n")
    print(embeddings)
    
    # Aplicar mean pooling
    mean_result = mean_pooling(embeddings)
    print("\n" + "="*70)
    print("MEAN POOLING")
    print("="*70)
    print("Resultado (média de cada dimensão):")
    print(mean_result)
    
    # Aplicar max pooling
    max_result = max_pooling(embeddings)
    print("\n" + "="*70)
    print("MAX POOLING")
    print("="*70)
    print("Resultado (máximo de cada dimensão):")
    print(max_result)
    
    # Calcular métricas de preservação de informação
    print("\n" + "="*70)
    print("ANÁLISE DE PRESERVAÇÃO DE INFORMAÇÃO")
    print("="*70)
    
    metricas_mean = calcular_preservacao_informacao(embeddings, mean_result, "Mean Pooling")
    metricas_max = calcular_preservacao_informacao(embeddings, max_result, "Max Pooling")
    
    print(f"\nMean Pooling:")
    print(f"  - Variância original: {metricas_mean['variancia_original']:.4f}")
    print(f"  - Variância do resultado: {metricas_mean['variancia_pooled']:.4f}")
    print(f"  - Norma do vetor: {metricas_mean['norma']:.4f}")
    print(f"  - Range de valores: [{metricas_mean['valores_min']:.4f}, {metricas_mean['valores_max']:.4f}]")
    
    print(f"\nMax Pooling:")
    print(f"  - Variância original: {metricas_max['variancia_original']:.4f}")
    print(f"  - Variância do resultado: {metricas_max['variancia_pooled']:.4f}")
    print(f"  - Norma do vetor: {metricas_max['norma']:.4f}")
    print(f"  - Range de valores: [{metricas_max['valores_min']:.4f}, {metricas_max['valores_max']:.4f}]")
    
    # Análise comparativa
    print("\n" + "="*70)
    print("CONCLUSÃO: QUAL ESTRATÉGIA PRESERVA MAIS INFORMAÇÃO?")
    print("="*70)
    
    print("\n📊 Mean Pooling:")
    print("  ✓ Preserva a distribuição geral dos valores")
    print("  ✓ Suaviza outliers e ruído")
    print("  ✓ Todos os tokens contribuem igualmente")
    print("  ✗ Perde informação sobre valores extremos")
    print("  ✗ Tokens com valores baixos podem diluir a informação")
    
    print("\n📊 Max Pooling:")
    print("  ✓ Preserva características salientes (valores máximos)")
    print("  ✓ Captura os valores mais importantes de cada dimensão")
    print("  ✓ Útil quando features específicas são mais relevantes")
    print("  ✗ Ignora completamente valores baixos e médios")
    print("  ✗ Sensível a outliers")
    
    print("\n🎯 Neste exemplo específico:")
    if metricas_max['variancia_pooled'] > metricas_mean['variancia_pooled']:
        print("  → Max Pooling preserva MAIS variância/diversidade no resultado")
        print("  → Melhor para capturar características distintas/salientes")
    else:
        print("  → Mean Pooling preserva MAIS a distribuição geral")
        print("  → Melhor para representação balanceada de toda a sequência")
    
    print("\n💡 Recomendação:")
    print("  A escolha depende da aplicação:")
    print("  - Mean Pooling: melhor para representação balanceada e robustez")
    print("  - Max Pooling: melhor para detecção de características salientes")
    
    # Criar visualizações
    print("\n" + "="*70)
    print("Gerando visualizações...")
    visualizar_comparacao(embeddings, mean_result, max_result)

if __name__ == "__main__":
    main()