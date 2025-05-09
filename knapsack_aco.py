import random
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class AntColonyOptimization:
    def __init__(self, 
                weights: List[int], 
                values: List[int], 
                capacity: int, 
                num_ants: int = 50, 
                max_iterations: int = 100,
                alpha: float = 1.0,  # Importância do feromônio
                beta: float = 2.0,   # Importância da heurística
                rho: float = 0.5,    # Taxa de evaporação
                q0: float = 0.9):    # Parâmetro de exploração/explotação
        """
        Inicializa o algoritmo ACO para o problema da mochila 0/1.
        
        Args:
            weights: Lista com os pesos dos itens
            values: Lista com os valores dos itens
            capacity: Capacidade máxima da mochila
            num_ants: Número de formigas
            max_iterations: Número máximo de iterações
            alpha: Importância do feromônio
            beta: Importância da informação heurística
            rho: Taxa de evaporação do feromônio
            q0: Probabilidade de escolha determinística
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_items = len(weights)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
        # Calcula razão valor/peso para heurística
        self.value_per_weight = [self.values[i] / self.weights[i] if self.weights[i] > 0 else 0 
                                for i in range(self.n_items)]
        
        # Inicializa matriz de feromônio (inicialmente igual para todos os itens)
        self.pheromone = np.ones(self.n_items)
        
        # Estatísticas
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Melhor solução global
        self.best_solution = None
        self.best_fitness = 0
    
    def generate_solution(self, ant_id: int) -> List[int]:
        """Gera uma solução para uma formiga"""
        solution = [0] * self.n_items
        remaining_capacity = self.capacity
        
        # Lista de itens ainda não considerados
        available_items = list(range(self.n_items))
        
        while available_items and remaining_capacity > 0:
            # Cálculo das probabilidades de seleção
            probabilities = []
            
            for item_idx in available_items:
                if self.weights[item_idx] <= remaining_capacity:
                    # Combina feromônio com informação heurística
                    tau = self.pheromone[item_idx] ** self.alpha
                    eta = self.value_per_weight[item_idx] ** self.beta
                    probabilities.append((item_idx, tau * eta))
                else:
                    probabilities.append((item_idx, 0))  # Sem chance de ser selecionado
            
            # Se não houver item válido, encerra a construção
            if not probabilities or all(p[1] == 0 for p in probabilities):
                break
            
            # Exploração vs explotação (regra pseudo-aleatória proporcional)
            q = random.random()
            if q < self.q0:  # Exploração determinística
                max_prob_item = max(probabilities, key=lambda x: x[1])
                selected_item = max_prob_item[0]
            else:  # Exploração probabilística
                # Normalizando probabilidades
                sum_prob = sum(p[1] for p in probabilities)
                if sum_prob == 0:
                    break
                
                norm_probabilities = [(p[0], p[1] / sum_prob) for p in probabilities]
                # Seleção baseada na roleta
                r = random.random()
                cumulative_prob = 0
                selected_item = None
                for item_idx, prob in norm_probabilities:
                    cumulative_prob += prob
                    if r <= cumulative_prob:
                        selected_item = item_idx
                        break
                
                # Trata caso de erro numérico
                if selected_item is None and norm_probabilities:
                    selected_item = norm_probabilities[-1][0]
            
            # Adiciona o item selecionado à solução
            solution[selected_item] = 1
            remaining_capacity -= self.weights[selected_item]
            available_items.remove(selected_item)
        
        return solution
    
    def fitness_function(self, solution: List[int]) -> Tuple[float, int, int]:
        """
        Calcula a aptidão de uma solução.
        
        Args:
            solution: Vetor binário representando os itens selecionados
            
        Returns:
            Tuple com (aptidão ajustada, valor total, peso total)
        """
        total_value = sum(self.values[i] for i in range(self.n_items) if solution[i] == 1)
        total_weight = sum(self.weights[i] for i in range(self.n_items) if solution[i] == 1)
        
        # Penaliza soluções que excedem a capacidade
        if total_weight > self.capacity:
            fitness = 0
        else:
            fitness = total_value
            
        return fitness, total_value, total_weight
    
    def update_pheromone(self, solutions: List[List[int]], fitness_values: List[float]):
        """Atualiza os níveis de feromônio com base nas soluções geradas"""
        # Evaporação do feromônio
        self.pheromone = (1 - self.rho) * self.pheromone
        
        # Deposição de feromônio
        for i, solution in enumerate(solutions):
            if fitness_values[i] > 0:  # Apenas soluções válidas depositam feromônio
                for j in range(self.n_items):
                    if solution[j] == 1:
                        self.pheromone[j] += fitness_values[i]
        
        # Limita feromônio para evitar estagnação
        self.pheromone = np.clip(self.pheromone, 0.1, 10.0)
    
    def find_best_solution(self) -> Dict:
        """Executa o algoritmo ACO e retorna a melhor solução encontrada"""
        start_time = time.time()
        
        # Iterações principais
        for iteration in range(self.max_iterations):
            solutions = []
            fitness_values = []
            
            # Cada formiga constrói uma solução
            for ant in range(self.num_ants):
                solution = self.generate_solution(ant)
                fitness, value, weight = self.fitness_function(solution)
                
                solutions.append(solution)
                fitness_values.append(fitness)
                
                # Atualiza melhor solução global
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution.copy()
            
            # Atualiza feromônio
            self.update_pheromone(solutions, fitness_values)
            
            # Registra estatísticas
            self.best_fitness_history.append(self.best_fitness)
            avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
            self.avg_fitness_history.append(avg_fitness)
        
        # Calcula estatísticas finais
        _, best_value, best_weight = self.fitness_function(self.best_solution)
        selected_items = [i for i in range(self.n_items) if self.best_solution[i] == 1]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "selected_items": selected_items,
            "selected_items_binary": self.best_solution,
            "total_value": best_value,
            "total_weight": best_weight,
            "execution_time": execution_time
        }
    
    def plot_progress(self):
        """Plota a evolução da aptidão média e da melhor aptidão por iteração"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'r-', label='Melhor Aptidão')
        plt.plot(self.avg_fitness_history, 'b-', label='Aptidão Média')
        plt.title('Evolução da Aptidão por Iteração (ACO)')
        plt.xlabel('Iteração')
        plt.ylabel('Aptidão')
        plt.legend()
        plt.grid(True)
        plt.savefig('aco_fitness_progress.png')
        plt.show()


def test_knapsack_example():
    """Teste com o exemplo fornecido na descrição do problema"""
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    
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
    
    print("\n--- Exemplo Pequeno (ACO) ---")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Representação binária: {result['selected_items_binary']}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    aco.plot_progress()


def test_medium_knapsack():
    """Teste com um exemplo de tamanho médio"""
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = [22, 33, 44, 55, 66, 77, 88, 99, 110]
    capacity = 200
    
    aco = AntColonyOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_ants=100,
        max_iterations=200,
        alpha=1.0,
        beta=2.0,
        rho=0.3,
        q0=0.9
    )
    
    result = aco.find_best_solution()
    
    print("\n--- Exemplo Médio (ACO) ---")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Representação binária: {result['selected_items_binary']}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    aco.plot_progress()


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
    
    aco = AntColonyOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_ants=100,
        max_iterations=100,  # Menos iterações para conjuntos grandes
        alpha=1.0,
        beta=3.0,  # Mais foco na heurística para problemas grandes
        rho=0.2,
        q0=0.9
    )
    
    result = aco.find_best_solution()
    
    print(f"\n--- Exemplo Grande ({n_items} itens) (ACO) ---")
    print(f"Número de itens selecionados: {len(result['selected_items'])}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    aco.plot_progress()


if __name__ == "__main__":
    # Testa exemplo pequeno (fornecido no problema)
    test_knapsack_example()
    
    # Testa exemplo de tamanho médio
    test_medium_knapsack()
    
    # Testa exemplo grande com 1000 itens
    test_large_knapsack(1000)
    
    # Descomente para testar com 10000 itens (pode levar mais tempo)
    # test_large_knapsack(10000) 