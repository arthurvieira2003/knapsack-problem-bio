import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from knapsack_genetic import GeneticAlgorithm
from knapsack_pso import ParticleSwarmOptimization
from knapsack_aco import AntColonyOptimization

# Para reprodutibilidade
random.seed(42)
np.random.seed(42)

def generate_dataset(n_items, max_weight=100, max_value=100):
    """Gera um conjunto de dados para o problema da mochila"""
    weights = [random.randint(1, max_weight) for _ in range(n_items)]
    values = [random.randint(1, max_value) for _ in range(n_items)]
    capacity = int(sum(weights) * 0.3)  # 30% da soma total dos pesos
    
    return weights, values, capacity

def run_comparison(weights, values, capacity, dataset_name=""):
    """Executa os três algoritmos e compara seus resultados"""
    n_items = len(weights)
    
    # Parâmetros comuns
    iterations = 100 if n_items <= 100 else 50
    population_size = 100 if n_items <= 100 else 200
    
    results = {}
    
    # Executa Algoritmo Genético
    print(f"\nExecutando Algoritmo Genético para {dataset_name}...")
    start_time = time.time()
    ga = GeneticAlgorithm(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=population_size,
        generations=iterations,
        crossover_rate=0.8,
        mutation_rate=0.1 if n_items <= 100 else 0.05
    )
    ga_result = ga.find_best_solution()
    ga_time = time.time() - start_time
    results["GA"] = {
        "valor": ga_result["total_value"],
        "peso": ga_result["total_weight"],
        "n_itens": len(ga_result["selected_items"]),
        "tempo": ga_time,
        "best_history": ga.best_fitness_history
    }
    
    # Executa PSO
    print(f"Executando PSO para {dataset_name}...")
    start_time = time.time()
    pso = ParticleSwarmOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_particles=population_size,
        max_iterations=iterations,
        w=0.5,
        c1=1.5,
        c2=1.5
    )
    pso_result = pso.find_best_solution()
    pso_time = time.time() - start_time
    results["PSO"] = {
        "valor": pso_result["total_value"],
        "peso": pso_result["total_weight"],
        "n_itens": len(pso_result["selected_items"]),
        "tempo": pso_time,
        "best_history": pso.best_fitness_history
    }
    
    # Executa ACO
    print(f"Executando ACO para {dataset_name}...")
    start_time = time.time()
    aco = AntColonyOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_ants=population_size,
        max_iterations=iterations,
        alpha=1.0,
        beta=2.0 if n_items <= 100 else 3.0,
        rho=0.5 if n_items <= 100 else 0.2,
        q0=0.9
    )
    aco_result = aco.find_best_solution()
    aco_time = time.time() - start_time
    results["ACO"] = {
        "valor": aco_result["total_value"],
        "peso": aco_result["total_weight"],
        "n_itens": len(aco_result["selected_items"]),
        "tempo": aco_time,
        "best_history": aco.best_fitness_history
    }
    
    # Imprime tabela comparativa
    print(f"\n{'=' * 60}")
    print(f"COMPARAÇÃO DE ALGORITMOS - {dataset_name}")
    print(f"{'=' * 60}")
    print(f"{'Algoritmo':<15} {'Valor':<10} {'Peso':<10} {'Itens':<10} {'Tempo (s)':<15}")
    print(f"{'-' * 60}")
    
    for alg, data in results.items():
        print(f"{alg:<15} {data['valor']:<10} {data['peso']:<10}/{capacity} {data['n_itens']:<10} {data['tempo']:<15.6f}")
    
    # Plota gráfico da evolução da aptidão
    plt.figure(figsize=(12, 7))
    plt.plot(results["GA"]["best_history"], 'r-', label='Algoritmo Genético')
    plt.plot(results["PSO"]["best_history"], 'g-', label='PSO')
    plt.plot(results["ACO"]["best_history"], 'b-', label='ACO')
    plt.title(f'Comparação da Evolução da Aptidão - {dataset_name}')
    plt.xlabel('Iteração')
    plt.ylabel('Melhor Aptidão')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'comparacao_{dataset_name.replace(" ", "_").lower()}.png')
    
    # Gráfico de barras para tempo de execução
    plt.figure(figsize=(10, 6))
    algorithms = list(results.keys())
    times = [results[alg]["tempo"] for alg in algorithms]
    values = [results[alg]["valor"] for alg in algorithms]
    
    ax1 = plt.subplot(1, 2, 1)
    bars = ax1.bar(algorithms, times, color=['red', 'green', 'blue'])
    ax1.set_ylabel('Tempo de Execução (s)')
    ax1.set_title('Tempo de Execução por Algoritmo')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona valores acima das barras
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom')
    
    ax2 = plt.subplot(1, 2, 2)
    bars = ax2.bar(algorithms, values, color=['red', 'green', 'blue'])
    ax2.set_ylabel('Valor Total Obtido')
    ax2.set_title('Valor Total por Algoritmo')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona valores acima das barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'metricas_{dataset_name.replace(" ", "_").lower()}.png')
    
    return results


def test_example_dataset():
    """Testa no exemplo fornecido no problema"""
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    
    return run_comparison(weights, values, capacity, "Exemplo Pequeno")
    

def test_medium_dataset():
    """Testa em um conjunto de dados médio"""
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = [22, 33, 44, 55, 66, 77, 88, 99, 110]
    capacity = 200
    
    return run_comparison(weights, values, capacity, "Exemplo Médio")


def test_large_dataset(n_items=1000):
    """Testa em um conjunto de dados grande"""
    weights, values, capacity = generate_dataset(n_items)
    
    return run_comparison(weights, values, capacity, f"Exemplo Grande ({n_items} itens)")


if __name__ == "__main__":
    # Teste com exemplo pequeno
    test_example_dataset()
    
    # Teste com exemplo médio
    test_medium_dataset()
    
    # Teste com exemplo grande (1000 itens)
    test_large_dataset(1000)
    
    # Teste com exemplo muito grande (10000 itens)
    # Descomente para executar (pode levar bastante tempo)
    # test_large_dataset(10000)
    
    plt.show()  # Mostra todos os gráficos no final 