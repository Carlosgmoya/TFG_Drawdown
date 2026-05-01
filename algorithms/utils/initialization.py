import numpy as np
import random
from .projection import projection_simplex

# This module contains functions to initialize the population of portfolios.

def initialize_population(N_pop, num_assets, cardinality):
    """
    Initialize the population with random portfolios.

    Returns:
    - population A: A 2D empty array
    - population B: A 2D array representing the initial population of portfolios.
    """
    population_A = np.empty((0, num_assets))
    population_B = []
    for _ in range(N_pop):
        individual = np.zeros(num_assets)
        selected_assets = random.sample(range(num_assets), random.randint(1, cardinality)) # Select random assets to include in the portfolio
        for asset in selected_assets:
            value = random .uniform(0, 1)
            while value == 0:
                value = random.uniform(0, 1)
            individual[asset] = value
        individual = projection_simplex(individual)
        population_B.append(individual)
        
    return population_A, np.array(population_B)