import random
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class GeneticAlgorithm:
    def __init__(self, 
                weights: List[int], 
                values: List[int], 
                capacity: int, 
                population_size: int = 100, 
                generations: int = 100,
                crossover_rate: float = 0.8,
                mutation_rate: float = 0.1):
        """
        Inicializa o Algoritmo Genético para o problema da mochila 0/1.
        
        Args:
            weights: Lista com os pesos dos itens
            values: Lista com os valores dos itens
            capacity: Capacidade máxima da mochila
            population_size: Tamanho da população
            generations: Número de gerações
            crossover_rate: Taxa de crossover
            mutation_rate: Taxa de mutação
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_items = len(weights)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Estatísticas
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self) -> List[List[int]]:
        """Inicializa a população com indivíduos aleatórios (representados como vetores binários)"""
        return [[random.randint(0, 1) for _ in range(self.n_items)] 
                for _ in range(self.population_size)]
    
    def fitness_function(self, individual: List[int]) -> Tuple[float, int, int]:
        """
        Calcula a aptidão de um indivíduo.
        
        Args:
            individual: Vetor binário representando os itens selecionados
            
        Returns:
            Tuple com (aptidão ajustada, valor total, peso total)
        """
        total_value = sum(self.values[i] for i in range(self.n_items) if individual[i] == 1)
        total_weight = sum(self.weights[i] for i in range(self.n_items) if individual[i] == 1)
        
        # Penaliza soluções que excedem a capacidade
        if total_weight > self.capacity:
            fitness = 0
        else:
            fitness = total_value
            
        return fitness, total_value, total_weight
    
    def selection(self, population: List[List[int]]) -> List[List[int]]:
        """Seleciona indivíduos para reprodução usando método de torneio"""
        tournament_size = 3
        selected = []
        
        for _ in range(self.population_size):
            # Seleciona k indivíduos aleatoriamente
            tournament = random.sample(population, tournament_size)
            # Avalia cada indivíduo
            tournament_fitness = [self.fitness_function(ind)[0] for ind in tournament]
            # Seleciona o melhor
            best_idx = tournament_fitness.index(max(tournament_fitness))
            selected.append(tournament[best_idx])
            
        return selected
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Realiza crossover de um ponto entre dois pais"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        crossover_point = random.randint(1, self.n_items - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutation(self, individual: List[int]) -> List[int]:
        """Aplica mutação com uma certa probabilidade em cada gene"""
        mutated = individual.copy()
        
        for i in range(self.n_items):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Inverte 0->1 ou 1->0
                
        return mutated
    
    def evolve(self, population: List[List[int]]) -> List[List[int]]:
        """Realiza uma geração completa: seleção, crossover e mutação"""
        # Seleção
        selected = self.selection(population)
        
        # Crossover e mutação
        offspring = []
        for i in range(0, self.population_size, 2):
            # Se atingimos o final com número ímpar
            if i + 1 >= self.population_size:
                offspring.append(self.mutation(selected[i]))
                continue
                
            # Crossover
            child1, child2 = self.crossover(selected[i], selected[i+1])
            
            # Mutação
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]  # Garante tamanho correto da população
    
    def find_best_solution(self) -> Dict:
        """Executa o algoritmo genético e retorna a melhor solução encontrada"""
        start_time = time.time()
        
        # Inicialização
        population = self.initialize_population()
        
        # Evolução
        best_individual = None
        best_fitness = -1
        
        for generation in range(self.generations):
            # Avaliação
            fitness_values = []
            for individual in population:
                fitness, _, _ = self.fitness_function(individual)
                fitness_values.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual
            
            # Registra estatísticas
            self.best_fitness_history.append(max(fitness_values))
            self.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
            
            # Evolução
            population = self.evolve(population)
        
        # Calcula estatísticas finais
        best_fitness, best_value, best_weight = self.fitness_function(best_individual)
        selected_items = [i for i in range(self.n_items) if best_individual[i] == 1]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "selected_items": selected_items,
            "selected_items_binary": best_individual,
            "total_value": best_value,
            "total_weight": best_weight,
            "execution_time": execution_time
        }
    
    def plot_progress(self):
        """Plota a evolução da aptidão média e da melhor aptidão por geração"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'r-', label='Melhor Aptidão')
        plt.plot(self.avg_fitness_history, 'b-', label='Aptidão Média')
        plt.title('Evolução da Aptidão por Geração')
        plt.xlabel('Geração')
        plt.ylabel('Aptidão')
        plt.legend()
        plt.grid(True)
        plt.savefig('fitness_progress.png')
        plt.show()


def test_knapsack_example():
    """Teste com o exemplo fornecido na descrição do problema"""
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    
    ga = GeneticAlgorithm(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    result = ga.find_best_solution()
    
    print("\n--- Exemplo Pequeno ---")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Representação binária: {result['selected_items_binary']}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    ga.plot_progress()


def test_medium_knapsack():
    """Teste com um exemplo de tamanho médio"""
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = [22, 33, 44, 55, 66, 77, 88, 99, 110]
    capacity = 200
    
    ga = GeneticAlgorithm(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=100,
        generations=200,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    result = ga.find_best_solution()
    
    print("\n--- Exemplo Médio ---")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Representação binária: {result['selected_items_binary']}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    ga.plot_progress()


def generate_large_dataset(n_items=1000):
    """Gera um conjunto de dados grande para teste"""
    random.seed(42)  # Para reprodutibilidade
    weights = [random.randint(1, 100) for _ in range(n_items)]
    values = [random.randint(1, 100) for _ in range(n_items)]
    capacity = int(sum(weights) * 0.3)  # 30% da soma total dos pesos
    
    return weights, values, capacity


def test_large_knapsack(n_items=1000):
    """Teste com um exemplo grande"""
    weights, values, capacity = generate_large_dataset(n_items)
    
    ga = GeneticAlgorithm(
        weights=weights,
        values=values,
        capacity=capacity,
        population_size=200,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.05
    )
    
    result = ga.find_best_solution()
    
    print(f"\n--- Exemplo Grande ({n_items} itens) ---")
    print(f"Número de itens selecionados: {len(result['selected_items'])}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    ga.plot_progress()


if __name__ == "__main__":
    # Testa exemplo pequeno (fornecido no problema)
    test_knapsack_example()
    
    # Testa exemplo de tamanho médio
    test_medium_knapsack()
    
    # Testa exemplo grande com 1000 itens
    test_large_knapsack(1000)
    
    # Descomente para testar com 10000 itens (pode levar mais tempo)
    # test_large_knapsack(10000) 