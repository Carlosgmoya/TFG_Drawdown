import time
import numpy as np
from scipy.spatial.distance import cdist
from algorithms.utils.initialization import initialize_population
from algorithms.utils.fitness import calculate_total_fitness
from algorithms.utils.operators import binary_tournament, crossover, mutation
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.normalization import normalize_objectives


class E_MOEA:
    def __init__(self, N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations, e):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns_matrix = returns_matrix # Historical returns matrix (n_assets, n_periods)
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of generations
        self.e = e # Epsilon value

        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)

        self.population_A = populations[0]
        self.population_B = populations[1]


    def is_not_dominated(self, ret_risk_B, matrix_ret_risks_A):
        """
        Check if an individual is not dominated by any individual in the population.
        
        Parameters:
        - ret_risk_B: (MDD, -mean_return) of the individual to check.
        - matrix_ret_risks_A: [MDD, -mean_return] of population A.
        
        Returns:
        - True if the individual is not dominated, False otherwise.
        """

        if len(matrix_ret_risks_A.T) == 0:
            return True

        mdd_B, neg_ret_B = ret_risk_B
        for ind2 in matrix_ret_risks_A.T:
            if self.dominates(ind2, (mdd_B, neg_ret_B)):
                return False
        return True


    def dominates_any(self, ret_risk_B, matrix_ret_risks_A):
        """
        Check if an individual dominates any individual in the population.
        
        Parameters:
        - ret_risk_B: (MDD, -mean_return) of the individual to check.
        - matrix_ret_risks_A: [MDD, -mean_return] of population A.
        
        Returns:
        - True if the individual dominates any, False otherwise.
        """
        
        if len(matrix_ret_risks_A.T) == 0:
            return True

        mdd_B, neg_ret_B = ret_risk_B
        for ind2 in matrix_ret_risks_A.T:
            if self.dominates((mdd_B, neg_ret_B), ind2):
                return True

        return False
    

    def smallest_distance_bigger_than_e(self, ind, points):
        """
        Check if the smallest distance between an individual and the population is > e.
        
        Parameters:
        - ind: (MDD, -mean_return) of the individual.
        - points: [MDD, -mean_return] of the population.
        
        Returns:
        - True if smallest distance > e, and the distances array.
        """
        
        if len(points.T) == 0:
            return False, np.array([])

        ind_array = np.array(ind)
        ind_array = ind_array[:, np.newaxis] 

        points = np.concatenate((points, ind_array), axis=1)
        points = (normalize_objectives(points)).T

        ind_obj = np.array([points[-1]])
        points = points[:-1]

        distances = cdist(ind_obj, points, 'cityblock')

        return np.min(distances[0]) > self.e, distances[0]


    def improve_optimum(self, evaluated_individual, points):
        """
        Check if the individual improves the optimum of a single objective.
        BOTH objectives are MINIMIZED: MDD and -mean_return.
        
        Parameters:
        - evaluated_individual: (MDD, -mean_return) of the individual.
        - points: Array of (MDD, -mean_return) for comparison.
        
        Returns:
        - True if it improves either objective's optimum.
        """

        if len(points) == 0:
            return False

        # Better if MDD is LOWER or -mean_return is LOWER
        if evaluated_individual[0] < np.min(points[:, 0]) or evaluated_individual[1] < np.min(points[:, 1]):
            return True
        return False


    def remove_dominated_solutions(self, population, matrix_ret_risks):
        """
        Remove dominated solutions from the population.
        
        Parameters:
        - population: The population.
        - matrix_ret_risks: [MDD, -mean_return] of the population.
        
        Returns:
        - non_dominated: Array of non-dominated solutions.
        """

        non_dominated = []
        for i in range(len(matrix_ret_risks.T)):
            if self.is_not_dominated(matrix_ret_risks.T[i], matrix_ret_risks):
                non_dominated.append(population[i])
        return np.array(non_dominated)


    def dominates(self, ret_risk1, ret_risk2):
        """
        Check if individual 1 dominates individual 2.
        BOTH objectives are MINIMIZED: MDD and -mean_return.
        
        Parameters:
        - ret_risk1: (MDD, -mean_return) of first individual.
        - ret_risk2: (MDD, -mean_return) of second individual.
        
        Returns:
        - True if first dominates second.
        """

        mdd1, neg_ret1 = ret_risk1
        mdd2, neg_ret2 = ret_risk2
        
        # Both minimized: <= in both, < in at least one
        return (mdd1 <= mdd2 and neg_ret1 <= neg_ret2) and (mdd1 < mdd2 or neg_ret1 < neg_ret2)
    

    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        fitness = calculate_total_fitness(population, self.returns_matrix)
        for _ in range(self.N_pop):
            parent1 = binary_tournament(population, fitness)
            parent2 = binary_tournament(population, fitness)
            while evaluate(parent1, self.returns_matrix) == evaluate(parent2, self.returns_matrix):
                parent2 = binary_tournament(population, fitness)
            child = crossover(parent1, parent2, self.num_assets, self.cardinality, self.crossover_rate)
            child = mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)


    def update(self, population_A, population_B):
        """
        Update the archive population with the current population.
        
        Parameters:
        - population_A: The archive population.
        - population_B: The current population.
        
        Returns:
        - population_A: The updated archive population.
        """
        
        matrix_ret_risks_A_new = precompute_objectives(population_A, self.returns_matrix)
        matrix_ret_risks_A_old = np.copy(matrix_ret_risks_A_new)

        for ind_b in population_B:
            added = False
            removed = False
            idxs_to_remove = []
            
            evaluate_b = evaluate(ind_b, self.returns_matrix)
            
            condition2, distances = self.smallest_distance_bigger_than_e(evaluate_b, matrix_ret_risks_A_new)

            # If it dominates any in old archive
            if self.dominates_any(evaluate_b, matrix_ret_risks_A_old):
                population_A = np.vstack((population_A, ind_b))
                added = True

            # If not dominated and far enough from existing solutions
            elif self.is_not_dominated(evaluate_b, matrix_ret_risks_A_old):
                if condition2:
                    population_A = np.vstack((population_A, ind_b))
                    added = True
            
            # If improves single-objective optimum
            if self.improve_optimum(evaluate_b, matrix_ret_risks_A_old.T):
                if not added:
                    population_A = np.vstack((population_A, ind_b))
                    added = True

                # Remove solutions too close to the new one
                for idx, dist in enumerate(distances):
                    if dist < self.e:
                        idxs_to_remove.append(idx)
                if len(idxs_to_remove) > 0:
                    population_A = np.delete(population_A, idxs_to_remove, axis=0)
                    removed = True

            if removed:
                matrix_ret_risks_A_new = np.delete(matrix_ret_risks_A_new, idxs_to_remove, axis=1)
            if added:
                matrix_ret_risks_A_new = np.vstack((matrix_ret_risks_A_new.T, evaluate_b)).T

        population_A = self.remove_dominated_solutions(population_A, matrix_ret_risks_A_new)

        return population_A


    def evolve(self):
        """
        Evolve the population using e-MOEA with MDD and -mean_return.
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