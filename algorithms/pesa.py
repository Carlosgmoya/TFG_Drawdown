import time
import numpy as np
from algorithms.utils.initialization import initialize_population
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.operators import crossover, mutation, binary_tournament


class PESA:
    def __init__(self, N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations, grid_divisions):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns_matrix = returns_matrix # Historical returns matrix (n_assets, n_periods)
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of generations
        self.grid_divisions = grid_divisions # Number of divisions in the grid

        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)
        self.population_A = populations[0] # Archive population (A_0)
        self.population_B = populations[1] # Usual population (B_0)


    def assign_hypergrid(self, points):
        """
        Assign points to a hypergrid.
        """
        # Limpiar todos los NaN de los puntos
        points_clean = np.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
        
        min_vals = np.min(points_clean, axis=0)
        max_vals = np.max(points_clean, axis=0)
        range_vals = max_vals - min_vals
        
        # Si no hay rango, todos los puntos van a la misma celda
        if range_vals[0] < 1e-10:
            range_vals[0] = 1.0
        if range_vals[1] < 1e-10:
            range_vals[1] = 1.0
        
        cell_size = range_vals / self.grid_divisions

        hypergrid = [[] for _ in range(self.grid_divisions * self.grid_divisions)]

        for idx, point in enumerate(points_clean):
            row = int((point[0] - min_vals[0]) / cell_size[0])
            col = int((point[1] - min_vals[1]) / cell_size[1])

            row = max(0, min(row, self.grid_divisions - 1))
            col = max(0, min(col, self.grid_divisions - 1))

            cell_index = row * self.grid_divisions + col
            hypergrid[cell_index].append(idx)

        return hypergrid     

    def get_individual_fitness(self, hypergrid, ind):
        """
        Get the fitness of an individual.
        
        Parameters:
        - hypergrid: The hypergrid.
        - ind: The individual.
        
        Returns:
        - length: The fitness of the individual (number in same cell).
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
        - matrix_ret_risks: [MDD, -mean_return] of the population.
        
        Returns:
        - fitness: The fitness of the population (higher = more crowded = worse).
        """

        points = matrix_ret_risks.T
        hypergrid = self.assign_hypergrid(points)
        fitness = np.zeros(len(matrix_ret_risks.T))

        for i in range(len(matrix_ret_risks.T)):
            fitness[i] = self.get_individual_fitness(hypergrid, i)

        return fitness


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.
        """
        new_population = []
        matrix_ret_risks = precompute_objectives(population, self.returns_matrix)
        fitness = self.fitness(matrix_ret_risks) 
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


    def is_not_dominated(self, ret_risk_B, matrix_ret_risks_A):
        """
        Check if an individual is not dominated by any individual in the population.
        
        Parameters:
        - ret_risk_B: (MDD, -mean_return) of the individual to check.
        - matrix_ret_risks_A: [MDD, -mean_return] of the archive population.
        
        Returns:
        - True if the individual is not dominated, False otherwise.
        """
        
        for ret_risk_A in matrix_ret_risks_A.T:
            if self.dominates(ret_risk_A, ret_risk_B):
                return False
        return True


    def dominates(self, ret_risk1, ret_risk2):
        """
        Check if individual 1 dominates individual 2.
        BOTH objectives are to be MINIMIZED: MDD and -mean_return.
        
        Parameters:
        - ret_risk1: (MDD, -mean_return) of the first individual.
        - ret_risk2: (MDD, -mean_return) of the second individual.
        
        Returns:
        - True if the first individual dominates the second, False otherwise.
        """

        mdd1, neg_ret1 = ret_risk1[0], ret_risk1[1]
        mdd2, neg_ret2 = ret_risk2[0], ret_risk2[1]
        
        # Both objectives to minimize: <= in both, < in at least one
        return (mdd1 <= mdd2 and neg_ret1 <= neg_ret2) and (mdd1 < mdd2 or neg_ret1 < neg_ret2)


    def update(self, population_A, population_B):
        """
        Update the archive population with the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - population_A: The updated archive population.
        """
        
        matrix_ret_risks_A = precompute_objectives(population_A, self.returns_matrix)
        
        for ind_b in population_B:
            ret_risk_B = evaluate(ind_b, self.returns_matrix)
            # If the individual is not dominated by any individual in the archive
            if self.is_not_dominated(ret_risk_B, matrix_ret_risks_A):
                # Remove individuals from archive that are dominated by the new one
                idxs_to_remove = []
                for idx, ret_risk_A in enumerate(matrix_ret_risks_A.T):
                    if self.dominates(ret_risk_B, ret_risk_A):
                        idxs_to_remove.append(idx)

                population_A = np.delete(population_A, idxs_to_remove, axis=0)
                matrix_ret_risks_A = np.delete(matrix_ret_risks_A.T, idxs_to_remove, axis=0).T

                # Add the individual to the archive
                population_A = np.vstack((population_A, ind_b))
                matrix_ret_risks_A = np.vstack((matrix_ret_risks_A.T, ret_risk_B)).T
                
                # If archive exceeds maximum size, remove most crowded
                if len(population_A) > self.N_arc:
                    fitness = self.fitness(matrix_ret_risks_A)
                    max_indices = np.where(fitness == np.max(fitness))[0]
                    to_remove = np.random.choice(max_indices)
                    population_A = np.delete(population_A, to_remove, axis=0)
                    matrix_ret_risks_A = np.delete(matrix_ret_risks_A.T, to_remove, axis=0).T

        return population_A


    def evolve(self):
        """
        Evolve the population using PESA with MDD and -mean_return.
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