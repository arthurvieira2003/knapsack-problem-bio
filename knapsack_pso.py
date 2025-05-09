import random
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class ParticleSwarmOptimization:
    def __init__(self, 
                weights: List[int], 
                values: List[int], 
                capacity: int, 
                num_particles: int = 50, 
                max_iterations: int = 100,
                w: float = 0.5,  # Inércia
                c1: float = 1.5,  # Coeficiente cognitivo
                c2: float = 1.5): # Coeficiente social
        """
        Inicializa o algoritmo PSO para o problema da mochila 0/1.
        
        Args:
            weights: Lista com os pesos dos itens
            values: Lista com os valores dos itens
            capacity: Capacidade máxima da mochila
            num_particles: Número de partículas no enxame
            max_iterations: Número máximo de iterações
            w: Fator de inércia
            c1: Fator de aprendizado cognitivo
            c2: Fator de aprendizado social
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_items = len(weights)
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Estatísticas
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize_particles(self):
        """Inicializa as partículas com posições e velocidades aleatórias"""
        # Posições: valores contínuos entre 0 e 1
        positions = np.random.random((self.num_particles, self.n_items))
        
        # Velocidades: valores entre -1 e 1
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.n_items))
        
        # Inicializa melhores posições pessoais (pbest)
        pbest_positions = positions.copy()
        pbest_fitness = np.zeros(self.num_particles)
        
        # Avalia fitness inicial
        for i in range(self.num_particles):
            binary_position = self.continuous_to_binary(positions[i])
            fitness, _, _ = self.fitness_function(binary_position)
            pbest_fitness[i] = fitness
        
        # Melhor posição global (gbest)
        gbest_idx = np.argmax(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        return positions, velocities, pbest_positions, pbest_fitness, gbest_position, gbest_fitness
    
    def continuous_to_binary(self, position):
        """Converte posições contínuas para valores binários usando sigmóide"""
        sigmoid = 1 / (1 + np.exp(-position))
        binary = [1 if random.random() < sigmoid[i] else 0 for i in range(self.n_items)]
        return binary
    
    def fitness_function(self, binary_solution: List[int]) -> Tuple[float, int, int]:
        """
        Calcula a aptidão de uma solução binária.
        
        Args:
            binary_solution: Vetor binário representando os itens selecionados
            
        Returns:
            Tuple com (aptidão ajustada, valor total, peso total)
        """
        total_value = sum(self.values[i] for i in range(self.n_items) if binary_solution[i] == 1)
        total_weight = sum(self.weights[i] for i in range(self.n_items) if binary_solution[i] == 1)
        
        # Penaliza soluções que excedem a capacidade
        if total_weight > self.capacity:
            fitness = 0
        else:
            fitness = total_value
            
        return fitness, total_value, total_weight
    
    def update_velocity_position(self, positions, velocities, pbest_positions, gbest_position):
        """Atualiza a velocidade e posição de cada partícula"""
        for i in range(self.num_particles):
            r1 = np.random.random(self.n_items)
            r2 = np.random.random(self.n_items)
            
            # Atualiza velocidade
            cognitive_component = self.c1 * r1 * (pbest_positions[i] - positions[i])
            social_component = self.c2 * r2 * (gbest_position - positions[i])
            
            velocities[i] = self.w * velocities[i] + cognitive_component + social_component
            
            # Limita velocidade entre -4 e 4 (evita valores extremos de sigmóide)
            velocities[i] = np.clip(velocities[i], -4, 4)
            
            # Atualiza posição
            positions[i] = positions[i] + velocities[i]
        
        return positions, velocities
    
    def find_best_solution(self) -> Dict:
        """Executa o algoritmo PSO e retorna a melhor solução encontrada"""
        start_time = time.time()
        
        # Inicialização
        positions, velocities, pbest_positions, pbest_fitness, gbest_position, gbest_fitness = self.initialize_particles()
        
        # Iterações principais
        for iteration in range(self.max_iterations):
            # Lista para armazenar fitness de todas as partículas nesta iteração
            current_fitness_values = []
            
            # Para cada partícula
            for i in range(self.num_particles):
                # Converte posição contínua para binária
                binary_position = self.continuous_to_binary(positions[i])
                
                # Avalia a posição atual
                fitness, _, _ = self.fitness_function(binary_position)
                current_fitness_values.append(fitness)
                
                # Atualiza melhor posição pessoal (pbest)
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                    
                    # Atualiza melhor posição global (gbest)
                    if fitness > gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = positions[i].copy()
            
            # Atualiza velocidades e posições
            positions, velocities = self.update_velocity_position(positions, velocities, pbest_positions, gbest_position)
            
            # Registra estatísticas
            self.best_fitness_history.append(gbest_fitness)
            self.avg_fitness_history.append(sum(current_fitness_values) / len(current_fitness_values))
        
        # Converte a melhor posição global para binária
        gbest_binary = self.continuous_to_binary(gbest_position)
        
        # Calcula estatísticas finais
        _, best_value, best_weight = self.fitness_function(gbest_binary)
        selected_items = [i for i in range(self.n_items) if gbest_binary[i] == 1]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "selected_items": selected_items,
            "selected_items_binary": gbest_binary,
            "total_value": best_value,
            "total_weight": best_weight,
            "execution_time": execution_time
        }
    
    def plot_progress(self):
        """Plota a evolução da aptidão média e da melhor aptidão por iteração"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'r-', label='Melhor Aptidão')
        plt.plot(self.avg_fitness_history, 'b-', label='Aptidão Média')
        plt.title('Evolução da Aptidão por Iteração (PSO)')
        plt.xlabel('Iteração')
        plt.ylabel('Aptidão')
        plt.legend()
        plt.grid(True)
        plt.savefig('pso_fitness_progress.png')
        plt.show()


def test_knapsack_example():
    """Teste com o exemplo fornecido na descrição do problema"""
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    
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
    
    print("\n--- Exemplo Pequeno (PSO) ---")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Representação binária: {result['selected_items_binary']}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    pso.plot_progress()


def test_medium_knapsack():
    """Teste com um exemplo de tamanho médio"""
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = [22, 33, 44, 55, 66, 77, 88, 99, 110]
    capacity = 200
    
    pso = ParticleSwarmOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_particles=100,
        max_iterations=200,
        w=0.5,
        c1=1.5,
        c2=1.5
    )
    
    result = pso.find_best_solution()
    
    print("\n--- Exemplo Médio (PSO) ---")
    print(f"Itens selecionados (índices): {result['selected_items']}")
    print(f"Representação binária: {result['selected_items_binary']}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    pso.plot_progress()


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
    
    pso = ParticleSwarmOptimization(
        weights=weights,
        values=values,
        capacity=capacity,
        num_particles=200,
        max_iterations=500,
        w=0.4,
        c1=1.5,
        c2=1.5
    )
    
    result = pso.find_best_solution()
    
    print(f"\n--- Exemplo Grande ({n_items} itens) (PSO) ---")
    print(f"Número de itens selecionados: {len(result['selected_items'])}")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Tempo de execução: {result['execution_time']:.6f} segundos")
    
    pso.plot_progress()


if __name__ == "__main__":
    # Testa exemplo pequeno (fornecido no problema)
    test_knapsack_example()
    
    # Testa exemplo de tamanho médio
    test_medium_knapsack()
    
    # Testa exemplo grande com 1000 itens
    test_large_knapsack(1000)
    
    # Descomente para testar com 10000 itens (pode levar mais tempo)
    # test_large_knapsack(10000) 