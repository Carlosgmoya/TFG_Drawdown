import time
import numpy as np
from scipy.spatial.distance import cdist
from algorithms.utils.initialization import initialize_population
from algorithms.utils.operators import binary_tournament, crossover, mutation
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.fitness import dominates, calculate_total_fitness

class SPEA2:
    def __init__(self, N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns_matrix = returns_matrix # Historical returns matrix (n_assets, n_periods)
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of generations
        
        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)
        self.population_A = populations[0] # Archive population (A_0)
        self.population_B = populations[1] # Usual population (B_0)


    def compute_dominance_matrix(self, matrix_ret_risks):
        """
        Compute the dominance matrix for the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing [MDD, -mean_return] of the population.
        
        Returns:
        - domination_matrix: A 2D array representing the dominance matrix.
        """
        
        N = matrix_ret_risks.shape[1] # Number of individuals
        domination_matrix = np.zeros((N, N), dtype=bool) # Boolean dominance matrix
        for i in range(N):
            for j in range(i + 1, N):
                if dominates(matrix_ret_risks, i, j): # If i dominates j
                    domination_matrix[i, j] = True
                elif dominates(matrix_ret_risks, j, i): # If j dominates i
                    domination_matrix[j, i] = True
        return domination_matrix


    def update(self, population_A, population_B):
        """
        Select the best individuals from archive and current populations.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - new_archive: The combined best population.
        """

        combined = np.vstack((population_A, population_B))
        fitness, matrix_ret_risks = calculate_total_fitness(combined, self.returns_matrix, return_matrix=True)

        dom_matrix = self.compute_dominance_matrix(matrix_ret_risks)
        non_dominated_indices = [i for i in range(len(combined)) if not np.any(dom_matrix[:, i])]
        new_archive = [combined[i] for i in non_dominated_indices]

        # If more than allowed, truncate
        if len(new_archive) > self.N_arc:
            new_archive = self.truncate(new_archive, self.N_arc)
        
        # If less, complete with best dominated
        elif len(new_archive) < self.N_arc:
            dominated_indices = [i for i in range(len(combined)) if i not in non_dominated_indices]
            dominated_sorted = sorted(dominated_indices, key=lambda i: fitness[i])
            
            i = 0
            while len(new_archive) < self.N_arc:
                new_archive.append(combined[dominated_sorted[i]])
                i += 1

        return new_archive

    def truncate(self, population, N_arc):
        """
        Truncate the population by removing the least diverse solutions.
        Version simplificada y robusta.
        """
        N = len(population)
        
        if N <= N_arc:
            return population
        
        matrix_ret_risks = precompute_objectives(population, self.returns_matrix)
        points = matrix_ret_risks.T
        
        # Limpiar puntos
        points = np.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calcular distancias
        distance_matrix = cdist(points, points)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Estrategia simple: eliminar los que tienen menor distancia al vecino mas cercano
        n_to_remove = N - N_arc
        removed = set()
        
        for _ in range(n_to_remove):
            # Encontrar el par mas cercano entre los no eliminados
            remaining = [i for i in range(N) if i not in removed]
            
            if len(remaining) < 2:
                break
            
            min_dist = np.inf
            to_remove = remaining[0]
            
            for i in remaining:
                # Distancia minima a cualquier otro individuo no eliminado
                dists = [distance_matrix[i, j] for j in remaining if j != i]
                if dists:
                    d = min(dists)
                    if d < min_dist:
                        min_dist = d
                        to_remove = i
            
            removed.add(to_remove)
        
        return [population[i] for i in range(N) if i not in removed]


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.
        """
        new_population = []
        fitness = calculate_total_fitness(population, self.returns_matrix)
        for _ in range(self.N_pop):
            parent1 = binary_tournament(population, fitness)
            parent2 = binary_tournament(population, fitness)
            
            # Evitar bucle infinito: maximo 10 intentos
            attempts = 0
            while attempts < 10:
                ret1 = evaluate(parent1, self.returns_matrix)
                ret2 = evaluate(parent2, self.returns_matrix)
                if abs(ret1[0] - ret2[0]) > 1e-6 or abs(ret1[1] - ret2[1]) > 1e-6:
                    break
                parent2 = binary_tournament(population, fitness)
                attempts += 1
            
            child = crossover(parent1, parent2, self.num_assets, self.cardinality, self.crossover_rate)
            child = mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)


    def evolve(self):
        """
        Evolve the population using SPEA2 with MDD and -mean_return.
        """

        i = 0
        started_time = time.time()
        while i < self.generations:
            if i % 10 == 0:
                print(f"Generation: {i}")
            self.population_A = self.update(self.population_A, self.population_B)
            self.population_B = self.vary(self.population_A)
            i += 1

        self.population_A = self.update(self.population_A, self.population_B)
        elapsed_time = time.time() - started_time
        print(f"Execution time: {elapsed_time:.3f} seconds")

        return self.population_A