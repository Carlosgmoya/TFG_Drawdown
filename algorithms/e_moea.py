import time
import numpy as np
from scipy.spatial.distance import cdist
from algorithms.utils.initialization import initialize_population
from algorithms.utils.fitness import calculate_total_fitness
from algorithms.utils.operators import binary_tournament, crossover, mutation
from algorithms.utils.evaluation import precompute_objectives, evaluate
from algorithms.utils.normalization import normalize_objectives


class E_MOEA:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations, e):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns = returns # Returns of the assets
        self.cov_matrix = cov_matrix # Covariance matrix of the assets
        self.crossover_rate = crossover_rate # Crossover rate
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of genetations
        self.e = e # Epsilon value

        populations = initialize_population(self.N_pop, self.num_assets, self.cardinality)

        self.population_A = populations[0]
        self.population_B = populations[1]


    def is_not_dominated(self, ret_risk_B, matrix_ret_risks_A):
        """
        Check if an individual is not dominated by any individual in the population.
        
        Parameters:
        - ret_risk_B: Returns and risk of the individual to check.
        - matrix_ret_risks_A: A 2D array containing the returns and risks of the population A.
        
        Returns:
        - True if the individual is not dominated, False otherwise.
        """

        # If the archive population is empty, the individual is not dominated
        if(len(matrix_ret_risks_A.T) == 0):
            return True

        ret_B, risk_B = ret_risk_B
        for ind2 in matrix_ret_risks_A.T:
            if self.dominates(ind2, (ret_B, risk_B)):
                return False
        return True


    def dominates_any(self, ret_risk_B, matrix_ret_risks_A):
        """
        Check if an individual dominates any individual in the population.
        
        Parameters:
        - ret_risk_B: Returns and risk of the individual to check.
        - matrix_ret_risks_A: A 2D array containing the returns and risks of the population A.
        
        Returns:
        - True if the individual dominates any individual in the population, False otherwise.
        """
        
        # If the archive population is empty, the individual dominates
        if(len(matrix_ret_risks_A.T) == 0):
            return True

        ret_B, risk_B = ret_risk_B
        for ind2 in matrix_ret_risks_A.T:
            if self.dominates((ret_B, risk_B), ind2):
                return True

        return False
    

    def smallest_distance_bigger_than_e(self, ind, points):
        """
        Check if the smallest distance between an individual and any individual in the population is bigger than e.
        
        Parameters:
        - ind: The individual to check.
        - points: A 2D array containing the returns and risks of the population.
        
        Returns:
        - True if the smallest distance is bigger than e, False otherwise.
        """
        
        if len(points.T) == 0:
            return False, np.array([])

        # Convert the individual to a numpy array
        ind_array = np.array(ind)
        ind_array = ind_array[:, np.newaxis] 

        # Add the individual to the points
        points = np.concatenate((points, ind_array), axis=1)
        # Normalize the points
        points = (normalize_objectives(points)).T

        # Get the last point
        ind_obj = np.array([points[-1]])
        # Remove the last point
        points = points[:-1]

        # Calculate the distances
        distances = cdist(ind_obj, points, 'cityblock')

        # Return if the smallest distance is bigger than e and the distances
        return np.min(distances[0]) > self.e, distances[0]


    def improve_optimum(self, evaluated_individual, points):
        """
        Check if the evaluated individual improves the optimum of a single objective function.
        
        Parameters:
        - evaluated_individual: The individual to check.
        - points: A 2D array containing the returns and risks of the population.
        
        Returns:
        - True if the evaluated individual improves the optimum of a single objective function, False otherwise.
        """

        if(len(points) == 0):
            return False

        # If the return is higher or the risk is lower
        if(evaluated_individual[0] > np.max(points[:, 0]) or evaluated_individual[1] < np.min(points[:, 1])):
            return True
        return False


    def remove_dominated_solutions(self, population, matrix_ret_risks):
        """
        Remove dominated solutions from the population.
        
        Parameters:
        - population: The population to remove dominated solutions from.
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - non_dominated: A list of non-dominated solutions.
        """

        non_dominated = []
        for i in range(len(matrix_ret_risks.T)):
            # If the individual is not dominated
            if self.is_not_dominated(matrix_ret_risks.T[i], matrix_ret_risks):
                non_dominated.append(population[i])
        return np.array(non_dominated)


    def dominates(self, ret_risk1, ret_risk2):
        """
        Check if an individual dominates another individual.
        
        Parameters:
        - ret_risk1: Returns and risk of the first individual.
        - ret_risk2: Returns and risk of the second individual.
        
        Returns:
        - True if the first individual dominates the second, False otherwise.
        """

        ret1, risk1 = ret_risk1
        ret2, risk2 = ret_risk2
        return (ret1 >= ret2 and risk1 <= risk2) and (ret1 > ret2 or risk1 < risk2)
    

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
            # Select two parents
            parent1 = binary_tournament(population, fitness)
            parent2 = binary_tournament(population, fitness)
            # If the parents are the same, select another parent
            while evaluate(parent1, self.returns, self.cov_matrix) == evaluate(parent2, self.returns, self.cov_matrix):
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
        

        # Precompute the objectives for the archive population
        matrix_ret_risks_A_new = precompute_objectives(population_A, self.returns, self.cov_matrix)
        
        # Copy the objectives for the old archive population
        matrix_ret_risks_A_old = np.copy(matrix_ret_risks_A_new)

        # For each individual in the current population
        for ind_b in population_B:
            added = False
            removed = False
            idxs_to_remove = []
            
            # Evaluate the individual
            evaluate_b = evaluate(ind_b, self.returns, self.cov_matrix)
            
            # Check if the smallest distance between the individual and any individual in the new archive population is bigger than e
            # And get the distances
            condition2, distances = self.smallest_distance_bigger_than_e(evaluate_b, matrix_ret_risks_A_new)

            # If the individual dominates any individual in the old archive population
            if self.dominates_any(evaluate_b, matrix_ret_risks_A_old):
                population_A = np.vstack((population_A, ind_b))
                added = True

            # If the individual is not dominated by any individual in the old archive population
            elif self.is_not_dominated(evaluate_b, matrix_ret_risks_A_old):
                if condition2:
                    population_A = np.vstack((population_A, ind_b))
                    added = True
            
            # If the individual improves the optimum of a single objective function on the old archive population
            if self.improve_optimum(evaluate_b, matrix_ret_risks_A_old.T):
                if not added:
                    population_A = np.vstack((population_A, ind_b))
                    added = True

                # Remove the individuals whose distance to the newly added individual is less than e
                for idx, dist in enumerate(distances):
                    if dist < self.e:
                        idxs_to_remove.append(idx)
                if len(idxs_to_remove) > 0:
                    population_A = np.delete(population_A, idxs_to_remove, axis=0)
                    removed = True

            # If an individual is added or removed, update the matrix of the new archive population
            if removed:
                matrix_ret_risks_A_new = np.delete(matrix_ret_risks_A_new, idxs_to_remove, axis=1)
            if added:
                matrix_ret_risks_A_new = np.vstack((matrix_ret_risks_A_new.T, evaluate_b)).T

        # Remove dominated solutions
        population_A = self.remove_dominated_solutions(population_A, matrix_ret_risks_A_new)

        return population_A


    def evolve(self):
        """
        Evolve the population over a number of generations using e-MOEA.

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

        self.population_A = self.update(self.population_A, self.population_A)
        elapsed_time = time.time() - started_time
        print(f"Execution time: {elapsed_time:.3f} seconds")

        return self.population_A