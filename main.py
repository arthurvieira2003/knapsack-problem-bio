import sys
import random
import numpy as np
import time

from knapsack_genetic import GeneticAlgorithm, test_knapsack_example as ga_test
from knapsack_pso import ParticleSwarmOptimization, test_knapsack_example as pso_test
from knapsack_aco import AntColonyOptimization, test_knapsack_example as aco_test
from comparacao_algoritmos import test_example_dataset, test_medium_dataset, test_large_dataset

def print_welcome():
    """Exibe mensagem de boas-vindas e opções"""
    print("\n" + "=" * 70)
    print("PROBLEMA DA MOCHILA COM ALGORITMOS BIO-INSPIRADOS".center(70))
    print("=" * 70)
    print("""
Este programa resolve o problema da mochila 0/1 usando algoritmos bio-inspirados.

Opções disponíveis:
1. Executar Algoritmo Genético (GA)
2. Executar Otimização por Enxame de Partículas (PSO)
3. Executar Otimização por Colônia de Formigas (ACO)
4. Comparar todos os algoritmos em exemplo pequeno
5. Comparar todos os algoritmos em exemplo médio
6. Comparar todos os algoritmos em exemplo grande (1000 itens)
7. Comparar todos os algoritmos em exemplo muito grande (10000 itens)
8. Inserir problema personalizado
0. Sair
    """)

def get_user_choice():
    """Obtém a escolha do usuário"""
    while True:
        try:
            choice = int(input("Escolha uma opção (0-8): "))
            if 0 <= choice <= 8:
                return choice
            print("Opção inválida. Tente novamente.")
        except ValueError:
            print("Por favor, digite um número válido.")

def custom_problem():
    """Permite ao usuário definir um problema personalizado"""
    print("\n--- Definição de Problema Personalizado ---")
    
    # Número de itens
    while True:
        try:
            n_items = int(input("Número de itens (1-10000): "))
            if 1 <= n_items <= 10000:
                break
            print("Número inválido. Tente novamente.")
        except ValueError:
            print("Por favor, digite um número válido.")
    
    # Geração aleatória ou entrada manual
    while True:
        gen_method = input("Gerar dados aleatoriamente? (s/n): ").lower()
        if gen_method in ["s", "n"]:
            break
        print("Entrada inválida. Digite 's' para sim ou 'n' para não.")
    
    if gen_method == "s":
        # Geração aleatória
        weights = [random.randint(1, 100) for _ in range(n_items)]
        values = [random.randint(1, 100) for _ in range(n_items)]
        
        # Capacidade como porcentagem do peso total
        while True:
            try:
                percent = float(input("Capacidade da mochila (% do peso total, entre 10-90): "))
                if 10 <= percent <= 90:
                    capacity = int(sum(weights) * (percent / 100))
                    break
                print("Porcentagem inválida. Tente novamente.")
            except ValueError:
                print("Por favor, digite um número válido.")
    else:
        # Entrada manual
        if n_items > 20:
            print("AVISO: Para muitos itens, a entrada manual pode ser tediosa.")
            confirm = input("Deseja continuar? (s/n): ").lower()
            if confirm != 's':
                return None, None, None
        
        weights = []
        values = []
        
        print("\nInsira os pesos e valores para cada item:")
        for i in range(n_items):
            while True:
                try:
                    weight = int(input(f"Peso do item {i+1}: "))
                    if weight > 0:
                        break
                    print("O peso deve ser maior que zero.")
                except ValueError:
                    print("Por favor, digite um número válido.")
            
            while True:
                try:
                    value = int(input(f"Valor do item {i+1}: "))
                    if value > 0:
                        break
                    print("O valor deve ser maior que zero.")
                except ValueError:
                    print("Por favor, digite um número válido.")
            
            weights.append(weight)
            values.append(value)
        
        # Capacidade
        while True:
            try:
                capacity = int(input("Capacidade da mochila: "))
                if capacity > 0:
                    break
                print("A capacidade deve ser maior que zero.")
            except ValueError:
                print("Por favor, digite um número válido.")
    
    # Escolha do algoritmo
    print("\nEscolha o algoritmo a ser executado:")
    print("1. Algoritmo Genético (GA)")
    print("2. Otimização por Enxame de Partículas (PSO)")
    print("3. Otimização por Colônia de Formigas (ACO)")
    print("4. Comparar todos os algoritmos")
    
    while True:
        try:
            alg_choice = int(input("Escolha (1-4): "))
            if 1 <= alg_choice <= 4:
                break
            print("Opção inválida. Tente novamente.")
        except ValueError:
            print("Por favor, digite um número válido.")
    
    # Executar o algoritmo escolhido
    if alg_choice == 1:
        run_ga(weights, values, capacity)
    elif alg_choice == 2:
        run_pso(weights, values, capacity)
    elif alg_choice == 3:
        run_aco(weights, values, capacity)
    elif alg_choice == 4:
        from comparacao_algoritmos import run_comparison
        run_comparison(weights, values, capacity, "Problema Personalizado")

def run_ga(weights=None, values=None, capacity=None):
    """Executa o Algoritmo Genético"""
    if weights is None:
        ga_test()
        return
    
    print("\nExecutando Algoritmo Genético...")
    start_time = time.time()
    
    ga = GeneticAlgorithm(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    result = ga.find_best_solution()
    
    print("\n--- Resultado do Algoritmo Genético ---")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Número de itens selecionados: {len(result['selected_items'])}")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Tempo de execução: {time.time() - start_time:.6f} segundos")
    
    ga.plot_progress()

def run_pso(weights=None, values=None, capacity=None):
    """Executa o PSO"""
    if weights is None:
        pso_test()
        return
    
    print("\nExecutando PSO...")
    start_time = time.time()
    
    pso = ParticleSwarmOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_particles=50,
        max_iterations=100,
        w=0.5,
        c1=1.5,
        c2=1.5
    )
    
    result = pso.find_best_solution()
    
    print("\n--- Resultado do PSO ---")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Número de itens selecionados: {len(result['selected_items'])}")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Tempo de execução: {time.time() - start_time:.6f} segundos")
    
    pso.plot_progress()

def run_aco(weights=None, values=None, capacity=None):
    """Executa o ACO"""
    if weights is None:
        aco_test()
        return
    
    print("\nExecutando ACO...")
    start_time = time.time()
    
    aco = AntColonyOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_ants=50,
        max_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        q0=0.9
    )
    
    result = aco.find_best_solution()
    
    print("\n--- Resultado do ACO ---")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Número de itens selecionados: {len(result['selected_items'])}")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Tempo de execução: {time.time() - start_time:.6f} segundos")
    
    aco.plot_progress()

def main():
    """Função principal"""
    while True:
        print_welcome()
        choice = get_user_choice()
        
        if choice == 0:
            print("Obrigado por usar o programa. Até logo!")
            sys.exit(0)
        elif choice == 1:
            run_ga()
        elif choice == 2:
            run_pso()
        elif choice == 3:
            run_aco()
        elif choice == 4:
            test_example_dataset()
        elif choice == 5:
            test_medium_dataset()
        elif choice == 6:
            test_large_dataset(1000)
        elif choice == 7:
            print("\nAVISO: A execução com 10000 itens pode levar muito tempo.")
            confirm = input("Deseja continuar? (s/n): ").lower()
            if confirm == 's':
                test_large_dataset(10000)
        elif choice == 8:
            custom_problem()
        
        input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    # Inicializa seeds para reprodutibilidade
    random.seed(42)
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário.")
        sys.exit(0) 