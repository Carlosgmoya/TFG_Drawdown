import random
import time
import numpy as np
from algorithms.utils.initialization import initialize_population
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.operators import crossover, mutation
from algorithms.utils.fitness import dominates

class NSGA2:
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


    def fast_non_dominated_sort(self, population):
        """
        Perform fast non-dominated sorting on the population.

        Parameters:
        - population: A 2D array representing the population of portfolios.

        Returns:
        - fronts: A list of fronts, where each front contains the indices of the individuals in that front.
        """
        
        fronts = [[]] # List of fronts
        population_size = len(population) # Size of the population 
        dominance_count = np.zeros(population_size) # Count of how many individuals dominate each individual
        dominated_sets = [[] for _ in range(population_size)] # List of dominated sets for each individual

        # Returns a matrix with [MDD, -mean_return] for each individual
        matrix_ret_risks = precompute_objectives(population, self.returns_matrix) 
        
        for i in range(population_size):
            for j in range(population_size):
                if dominates(matrix_ret_risks, i, j): # If i dominates j
                    dominated_sets[i].append(j)
                elif dominates(matrix_ret_risks, j, i): # If j dominates i
                    dominance_count[i] += 1
            if dominance_count[i] == 0: # If no one dominates this individual, it belongs to the first front
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = [] # Next front to be filled
            for p in fronts[i]: # For each individual in the current front
                for q in dominated_sets[p]: # For each individual that individual p dominates
                    dominance_count[q] -= 1
                    if dominance_count[q] == 0: # If this individual is no longer dominated
                        next_front.append(q)
            i += 1
            fronts.append(next_front) # Add the next front to the list of fronts

        return fronts[:-1] # Remove the last empty front
    

    def crowding_distance_assignment(self, front, population):
        """
        Assign crowding distance to each individual in the front.

        Parameters:
        - front: A list of indices representing the individuals in the front.
        - population: A 2D array representing the population of portfolios.

        Returns:
        - distances: A 1D array representing the crowding distance of each individual in the front.
        """

        distances = np.zeros(len(front)) # Initialize distances to zero
        for m in range(2): # For each objective (MDD and -mean_return)
            front.sort(key=lambda x: evaluate(population[x], self.returns_matrix)[m]) # Sort by m-th objective
            distances[0] = distances[-1] = np.inf # Infinite distance to extremes
            min_val = evaluate(population[front[0]], self.returns_matrix)[m]
            max_val = evaluate(population[front[-1]], self.returns_matrix)[m]
            for i in range(1, len(front) - 1): # For each individual (except extremes)
                if max_val - min_val == 0: # Avoid division by zero
                    distances[i] = 0
                else:
                    # Calculate crowding distance: normalized distance between neighbors
                    distances[i] += (evaluate(population[front[i + 1]], self.returns_matrix)[m] -
                                     evaluate(population[front[i - 1]], self.returns_matrix)[m]) / (max_val - min_val)
                 
        return distances


    def selection(self, fronts, population, population_size):
        """
        Select individuals for the next generation based on non-dominated sorting and crowding distance.
        
        Parameters:
        - fronts: A list of fronts with indices.
        - population: A 2D array of portfolios.
        - population_size: The size of the objective population.

        Returns:
        - new_population: Selected individuals for the next generation.
        """

        new_population = [] # Initialize new population (indices)
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                distances = self.crowding_distance_assignment(front, population)
                sorted_front = sorted(zip(front, distances), key=lambda x: -x[1]) # Sort by distance (descending)
                new_individuals = sorted_front[:population_size - len(new_population)]
                new_population.extend([x[0] for x in new_individuals])
        return np.array([population[i] for i in new_population])


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        fronts = self.fast_non_dominated_sort(population)

        distances = [] # List of crowding distances for each front
        for front in fronts:
            distances.append(self.crowding_distance_assignment(front, population))
        
        for _ in range(self.N_pop):
            parents = self.binary_tournament(population, fronts, distances)
            child = crossover(population[parents[0]], population[parents[1]], self.num_assets, self.cardinality, self.crossover_rate)
            child = mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)
    

    def binary_tournament(self, population, fronts, crowding_distance):
        """
        Perform a binary tournament to select two parents.
        
        Parameters:
        - population: The current population.
        - fronts: A list of fronts with indices.
        - crowding_distance: A list of crowding distances for each front.
        
        Returns:
        - selected_parents: The selected individuals.
        """

        selected_parents = []
        for _ in range(2):
            candidates = random.sample(range(len(population)), 2)
            front_candidate1 = self.getIndexFront(fronts, candidates[0])
            front_candidate2 = self.getIndexFront(fronts, candidates[1])
            
            pos_in_front_ind1 = fronts[front_candidate1].index(candidates[0])
            pos_in_front_ind2 = fronts[front_candidate2].index(candidates[1])
            
            if front_candidate1 == front_candidate2: # Same front
                if crowding_distance[front_candidate1][pos_in_front_ind1] > crowding_distance[front_candidate2][pos_in_front_ind2]:
                    selected_parents.append(candidates[0])
                else:
                    selected_parents.append(candidates[1])
            else:
                if front_candidate1 < front_candidate2: # Better front (lower index)
                    selected_parents.append(candidates[0])
                else:
                    selected_parents.append(candidates[1])
        return np.array(selected_parents)


    def getIndexFront(self, front, index):
        """
        Get the index of the front that contains the given index.
        """
        for i, subarray in enumerate(front):
            if index in subarray:
                return i
        return -1


    def update(self, population_A, population_B):
        """
        Select the best individuals from archive and current populations.
        """
        combined_population = np.vstack((population_A, population_B))
        fronts = self.fast_non_dominated_sort(combined_population)
        selected_population = self.selection(fronts, combined_population, self.N_arc)
        return selected_population


    def evolve(self):
        """
        Evolve the population using NSGA-II with MDD and -mean_return as objectives.
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