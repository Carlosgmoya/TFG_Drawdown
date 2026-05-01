import time
import numpy as np
import random
from scipy.spatial.distance import cdist
from algorithms.utils.initialization import initialize_population
from algorithms.utils.fitness import dominates
from algorithms.utils.operators import binary_tournament, crossover, mutation
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.normalization import normalize_objectives


class NPGA2:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations, tdom, rsh):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns = returns # Returns of the assets
        self.cov_matrix = cov_matrix # Covariance matrix of the assets
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of genetations
        self.tdom = tdom  # Tournament size
        self.rsh = rsh    # Niche radius

        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)
        self.population_A = populations[0] # Archive population (A_0)
        self.population_B = populations[1] # Usual population (B_0)


    def dominance_rank(self, matrix_ret_risks):
        """
        Calculate the dominance rank of each individual in the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - rank: A 1D array containing the dominance rank of each individual.
        """

        N = matrix_ret_risks.shape[1] # Number of individuals
        rank = np.zeros(N) # Dominance rank of each individual
        for i in range(N):
            for j in range(N):
                if i != j:
                    if dominates(matrix_ret_risks, j, i):
                        rank[i] += 1
        return rank


    def manhattan_distance(self, matrix_ret_risks):
        """
        Calculate the Manhattan distance between individuals.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - distances: A 2D array containing the Manhattan distances between individuals.
        """

        norm_matrix = normalize_objectives(matrix_ret_risks).T
        distances = cdist(norm_matrix, norm_matrix, metric='cityblock')
        return distances


    def niche_count(self, distances, next_gen, i):
        """
        Calculate the niche count of an individual.
        
        Parameters:
        - distances: A 2D array containing the Manhattan distances between individuals.
        - next_gen: The next generation of individuals.
        - i: Index of the individual.
        
        Returns:
        - count: The niche count of the individual.
        """

        count = 0
        for j in range(len(next_gen)):
            if distances[i, j] < self.rsh: # If the distance is less than the niche radius
                count += 1 - distances[i, j] / self.rsh # Add the niche count
        return count


    def update(self, population_A, population_B):
        """
        Update the archive population.
        
        Parameters:
        - population_A: The archive population.
        - population_B: The current population.
        
        Returns:
        - archive: The updated archive population.
        """

        combined = np.vstack((population_A, population_B))
        matrix_ret_risks = precompute_objectives(combined, self.returns, self.cov_matrix)
        rank = self.dominance_rank(matrix_ret_risks)
        distances = self.manhattan_distance(matrix_ret_risks)

        archive = []
        available = set(range(len(combined))) # Set of available individuals
        while len(archive) < self.N_arc:
            if (len(available) < self.tdom): # If there are less available individuals than the tournament size
                candidates = random.sample(list(available), len(available))
            else:
                candidates = random.sample(list(available), self.tdom) # Sample a random subset of the available individuals
            best_rank = min(rank[i] for i in candidates) # Best rank
            best = [i for i in candidates if rank[i] == best_rank] # Best individuals
            if len(best) > 1: # If there are multiple best individuals
                niche_counts = [self.niche_count(distances, archive, i) for i in best] # Niche counts
                best_idx = best[np.argmin(niche_counts)] # Best individual, the one with the lowest niche count
            else:
                best_idx = best[0] # Only one individual

            archive.append(combined[best_idx]) # Add the best individual to the archive
            available.remove(best_idx) # Remove the best individual from the available set
        return archive


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.
        
        Parameters:
        - population: A 2D array representing the population of portfolios.
        
        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        ranks = self.dominance_rank(precompute_objectives(population, self.returns, self.cov_matrix))
        for _ in range(self.N_pop):
            parent1 = binary_tournament(population, ranks)
            parent2 = binary_tournament(population, ranks)
            # If the parents are the same, select another parent
            while evaluate(parent1, self.returns, self.cov_matrix) == evaluate(parent2, self.returns, self.cov_matrix):
                parent2 = binary_tournament(population, ranks)
            child = crossover(parent1, parent2, self.num_assets, self.cardinality, self.crossover_rate)
            child = mutation(child, self.mutation_rate)
            new_population.append(child)
        return new_population


    def evolve(self):
        """
        Evolve the population over a number of generations using NPGA2.

        Returns:
        - population: The final population after evolution.
        """

        t = 0
        started_time = time.time()
        while t < self.generations:
            if t % 10 == 0:
                print(f"Generation: {t}")
            self.population_A = self.update(self.population_A, self.population_B)
            self.population_B = self.vary(self.population_A)
            t += 1
        self.population_A = self.update(self.population_A, self.population_B)
        elapsed_time = time.time() - started_time
        print(f"Execution time: {elapsed_time:.3f} seconds")
        return self.population_A
