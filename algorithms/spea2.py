import time
import numpy as np
from scipy.spatial.distance import cdist
from algorithms.utils.initialization import initialize_population
from algorithms.utils.operators import binary_tournament, crossover, mutation
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.fitness import dominates, calculate_total_fitness

class SPEA2:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns = returns # Returns of the assets
        self.cov_matrix = cov_matrix # Covariance matrix of the assets
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of genetations
        
        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)
        self.population_A = populations[0] # Archive population (A_0)
        self.population_B = populations[1] # Usual population (B_0)


    def compute_dominance_matrix(self, matrix_ret_risks):
        """
        Compute the dominance matrix for the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - domination_matrix: A 2D array representing the dominance matrix.
        """
        
        N = matrix_ret_risks.shape[1] # Number of individuals
        domination_matrix = np.zeros((N, N), dtype=bool) # Boolean dominance matrix
        for i in range(N): # Index of the first individual
            for j in range(i + 1, N): # Index of the second individual
                if dominates(matrix_ret_risks, i, j): # If i dominates j
                    domination_matrix[i, j] = True
                elif dominates(matrix_ret_risks, j, i): # If j dominates i
                    domination_matrix[j, i] = True
        return domination_matrix


    def update(self, population_A, population_B):
        """
        Select the best individuals from the archive population and the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - new_archive: The combined best population from both populations.

        """

        combined = np.vstack((population_A, population_B)) # Combined population
        fitness, matrix_ret_risks = calculate_total_fitness(combined, self.returns, self.cov_matrix, return_matrix=True) # Fitness values


        dom_matrix = self.compute_dominance_matrix(matrix_ret_risks) # Dominance matrix
        non_dominated_indices = [i for i in range(len(combined)) if not np.any(dom_matrix[:, i])] # Non-dominated indices
        new_archive = [combined[i] for i in non_dominated_indices] # Non-dominated population

        # If there are more than allowed, truncate
        if len(new_archive) > self.N_arc:
            new_archive = self.truncate(new_archive, self.N_arc) # Truncate the population
        
        # If there are less, complete with the best dominated
        elif len(new_archive) < self.N_arc:
            dominated_indices = [i for i in range(len(combined)) if i not in non_dominated_indices] # Dominated indices
            dominated_sorted = sorted(dominated_indices, key=lambda i: fitness[i]) # Sort the dominated indices by fitness
            
            # Complete with the best dominated
            i = 0
            while len(new_archive) < self.N_arc:
                new_archive.append(combined[dominated_sorted[i]])
                i += 1

        return new_archive


    def truncate(self, population, N_arc):
        """
        Truncate the population by removing the least diverse solutions.

        Parameters:
        - population: The population to truncate.
        - N_arc: The target size of the population.

        Returns:
        - truncated_population: The truncated population.
        """

        N = len(population)
        matrix_ret_risks = precompute_objectives(population, self.returns, self.cov_matrix)
        points = matrix_ret_risks.T  # shape (N, 2)

        # Compute pairwise distances
        distance_matrix = cdist(points, points)
        np.fill_diagonal(distance_matrix, np.inf) # Set diagonal to infinity to avoid self-comparison

        # Indices of individuals still in the archive
        remaining = list(range(N))
        sort_distance_matrix = np.sort(distance_matrix, axis=1)
        nearest_distances = sort_distance_matrix[:, 0] # Nearest distances

        while len(remaining) > N_arc: # While there are more individuals than allowed
            min_indexes = np.where(nearest_distances == np.min(nearest_distances))[0] # Indices of the individuals with the minimum distance
            to_remove = None
            if(len(min_indexes) > 1): # If there are more than one individual with the minimum distance
                k = 1
                
                while to_remove is None:
                    if(k >= N): # If we have checked all individuals, remove the first one and exit the loop
                        to_remove = min_indexes[0]
                        continue

                    index_value = np.column_stack((min_indexes, sort_distance_matrix[min_indexes, k])) # Stack the indices with their distances
                    min_value = np.min(index_value[:, 1]) # Minimum distance in the next column
                    min_index2 = np.where(index_value[:, 1] == min_value)[0] # Indices of the individuals with the minimum distance in the next column

                    # If there is only one individual, remove it and exit the loop
                    if(len(min_index2) == 1): 
                        to_remove = index_value[min_index2[0]][0]
                        continue

                    # If there are more than one individual, continue with the next column
                    k += 1

            else: # If there is only one individual with the minimum distance
                to_remove = min_indexes[0]

            nearest_distances[int(to_remove)] = np.inf # Set the nearest distance of the removed individual to infinity, to avoid comparing it again
            remaining.remove(int(to_remove)) # Remove the individual from the remaining list
        
        return [population[i] for i in remaining] # Return the remaining population


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        fitness = calculate_total_fitness(population, self.returns, self.cov_matrix)
        for _ in range(self.N_pop):
            parent1 = binary_tournament(population, fitness)
            parent2 = binary_tournament(population, fitness)
            # If the parents are the same, select another parent
            while evaluate(parent1, self.returns, self.cov_matrix) == evaluate(parent2, self.returns, self.cov_matrix):
                parent2 = binary_tournament(population, fitness)
            child = crossover(parent1, parent2, self.num_assets, self.cardinality, self.crossover_rate)
            child = mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)


    def evolve(self):
        """
        Evolve the population over a number of generations using SPEA2.

        Returns:
        - population: The final population after evolution.
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