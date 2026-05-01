import time
import numpy as np
from algorithms.utils.initialization import initialize_population
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.operators import crossover, mutation, binary_tournament


class PESA:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations, grid_divisions):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns = returns # Returns of the assets
        self.cov_matrix = cov_matrix # Covariance matrix of the assets
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of genetations
        self.grid_divisions = grid_divisions # Number of divisions in the grid

        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)
        self.population_A = populations[0] # Archive population (A_0)
        self.population_B = populations[1] # Usual population (B_0)


    def assign_hypergrid(self, points):
        """
        Assign points to a hypergrid.
        
        Parameters:
        - points: A numpy array of shape (N, 2), where each point is (return, risk)
        
        Returns:
        - hypergrid: A list of lists with point indices assigned to grid cells
        """

        min_vals = np.min(points, axis=0) # Minimum values of returns and risks
        max_vals = np.max(points, axis=0) # Maximum values of returns and risks
        cell_size = (max_vals - min_vals) / self.grid_divisions # Size of each cell

        hypergrid = [[] for _ in range(self.grid_divisions * self.grid_divisions)] # Hypergrid

        for idx, point in enumerate(points):
            row = int((point[0] - min_vals[0]) / cell_size[0]) # Row of the cell
            col = int((point[1] - min_vals[1]) / cell_size[1]) # Column of the cell

            # Clamp to avoid going out of bounds due to precision
            row = min(row, self.grid_divisions - 1) 
            col = min(col, self.grid_divisions - 1)

            cell_index = row * self.grid_divisions + col # Index of the cell
            hypergrid[cell_index].append(idx) # Append the index of the point to the cell

        return hypergrid


    def get_individual_fitness(self, hypergrid, ind):
        """
        Get the fitness of an individual.
        
        Parameters:
        - hypergrid: The hypergrid.
        - ind: The individual.
        
        Returns:
        - length: The fitness of the individual.
        """

        length = 0
        for sublist in hypergrid:
            if ind in sublist:
                length = len(sublist)
                break
        return length


    def fitness(self, matrix_ret_risks):
        """
        Compute the fitness of the population, based on the number of individuals in each 
        cell of the hypergrid.
        
        Parameters:
        - matrix_ret_risks: The returns and risks of the population.
        
        Returns:
        - fitness: The fitness of the population.
        """

        points = matrix_ret_risks.T
        hypergrid = self.assign_hypergrid(points) # Hypergrid
        fitness = np.zeros(len(matrix_ret_risks.T))

        for i in range(len(matrix_ret_risks.T)):
            fitness[i] = self.get_individual_fitness(hypergrid, i) # Fitness of each individual

        return fitness


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        matrix_ret_risks = precompute_objectives(population, self.returns, self.cov_matrix)
        fitness = self.fitness(matrix_ret_risks) 
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


    def is_not_dominated(self, ret_risk_B, matrix_ret_risks_A):
        """
        Check if an individual is not dominated by any individual in the population.
        
        Parameters:
        - ret_risk_B: The returns and risk of the individual to check.
        - matrix_ret_risks_A: The returns and risks of the archive population.
        
        Returns:
        - True if the individual is not dominated, False otherwise.
        """
        
        for ret_risk_A in matrix_ret_risks_A.T:
            if self.dominates(ret_risk_A, ret_risk_B):
                return False
        return True


    def dominates(self, ret_risk1, ret_risk2):
        """
        Check if an individual dominates another individual.
        
        Parameters:
        - ret_risk1: Returns and risk of the first individual.
        - ret_risk2: Returns and risk of the second individual.
        
        Returns:
        - True if the first individual dominates the second, False otherwise.
        """

        return ret_risk1[0] >= ret_risk2[0] and ret_risk1[1] <= ret_risk2[1] and (ret_risk1[0] > ret_risk2[0] or ret_risk1[1] < ret_risk2[1])


    def update(self, population_A, population_B):
        """
        Update the archive population with the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - population_A: The updated archive population.
        """
        
        matrix_ret_risks_A = precompute_objectives(population_A, self.returns, self.cov_matrix)
        
        for ind_b in population_B:
            ret_risk_B = evaluate(ind_b, self.returns, self.cov_matrix)
            # If the individual is not dominated by any individual in the archive population
            if (self.is_not_dominated(ret_risk_B, matrix_ret_risks_A)):
                # If the individual dominates any individual in the archive population, will be removed
                idxs_to_remove = []
                for idx,ret_risk_A in enumerate(matrix_ret_risks_A.T):
                    if self.dominates(ret_risk_B, ret_risk_A):
                        idxs_to_remove.append(idx)

                population_A = np.delete(population_A, idxs_to_remove, axis=0)
                matrix_ret_risks_A = np.delete(matrix_ret_risks_A.T, idxs_to_remove, axis=0).T

                # Add the individual to the archive population
                population_A = np.vstack((population_A, ind_b))
                matrix_ret_risks_A = np.vstack((matrix_ret_risks_A.T, ret_risk_B)).T
                if len(population_A) > self.N_arc:
                    # Remove the individual with the highest fitness, if there are multiple individuals with
                    # the same fitness, remove one at random
                    fitness = self.fitness(matrix_ret_risks_A)
                    max_indices = np.where(fitness == np.max(fitness))[0]
                    to_remove = np.random.choice(max_indices)
                    population_A = np.delete(population_A, to_remove, axis=0)
                    matrix_ret_risks_A = np.delete(matrix_ret_risks_A.T, to_remove, axis=0).T

        return population_A


    def evolve(self):
        """
        Evolve the population over a number of generations using PESA.

        Returns:
        - population_A: The final population after evolution.
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