import numpy as np
from scipy.spatial.distance import cdist
from .evaluation import precompute_objectives

# This module contains functions to calculate the fitness of a population.

def dominates(matrix_ret_risks, ind1, ind2):
    """
    Check if individual 1 dominates individual 2 in MDD vs -Mean_Return space.
    
    Both objectives are to be MINIMIZED:
    - obj[0] = MDD (minimize)
    - obj[1] = -mean_return (minimize, equivalent to maximizing return)
    
    Parameters:
    - matrix_ret_risks: A 2D array of shape (2, N) where:
        matrix_ret_risks[0, :] = MDD values
        matrix_ret_risks[1, :] = -mean_return values
    - ind1: Index of first individual.
    - ind2: Index of second individual.
    
    Returns:
    - True if ind1 dominates ind2, False otherwise.
    """
    mdd1 = matrix_ret_risks[0][ind1]
    neg_ret1 = matrix_ret_risks[1][ind1]  # -mean_return (minimize)
    mdd2 = matrix_ret_risks[0][ind2]
    neg_ret2 = matrix_ret_risks[1][ind2]
    
    # Dominance: ind1 is better if it has <= in both and < in at least one
    # (remember: both objectives are to minimize)
    return (mdd1 <= mdd2 and neg_ret1 <= neg_ret2) and (mdd1 < mdd2 or neg_ret1 < neg_ret2)


def raw_fitness(population, matrix_ret_risks):
    """
    Calculate the raw fitness values for each individual in the population.
    Raw fitness = number of individuals that dominate it.
    
    Parameters:
    - population: A 2D array representing the population of portfolios.
    - matrix_ret_risks: A 2D array containing [MDD, -mean_return] for each individual.
    
    Returns:
    - raw_fitness_values: A 1D array with raw fitness (lower is better, 0 = non-dominated).
    """
    N = len(population)
    dominance_count = np.zeros(N)  # Number of individuals that dominate each individual
    dominated_sets = [[] for _ in range(N)]  # Individuals dominated by each individual
    
    for i in range(N):
        for j in range(N):
            if dominates(matrix_ret_risks, i, j):  # If i dominates j
                dominated_sets[i].append(j)
            elif dominates(matrix_ret_risks, j, i):  # If j dominates i
                dominance_count[i] += 1

    raw_fitness_values = np.zeros(N)
    for i in range(N):
        # Raw fitness: number of dominators + small penalty from dominated count
        raw_fitness_values[i] = dominance_count[i] / (len(dominated_sets[i]) if len(dominated_sets[i]) > 0 else 1)

    return raw_fitness_values


def calculate_density(matrix_ret_risks, k=1):
    """
    Calculate the density of each individual based on distance to k-th nearest neighbor.
    
    Parameters:
    - matrix_ret_risks: A 2D array of shape (2, N) with [MDD, -mean_return].
    - k: Number of neighbors to consider.
    
    Returns:
    - D: A 1D array with density values (higher = more crowded).
    """
    points = matrix_ret_risks.T  # Shape (N, 2) 
    distances = cdist(points, points)  # Pairwise distances
    np.fill_diagonal(distances, np.inf)  # Ignore self-distance
    kth_distances = np.partition(distances, k, axis=1)[:, k]  # k-th smallest distance
    return 1.0 / (kth_distances + 2.0)  # Density = inverse distance


def calculate_total_fitness(population, returns_matrix, k=1, return_matrix=False):
    """
    Calculate total fitness = raw fitness + density for SPEA2 algorithm.
    
    Parameters:
    - population: A 2D array representing the population of portfolios.
    - returns_matrix: Historical returns matrix (n_assets, n_periods).
    - k: Number of neighbors for density estimation.
    - return_matrix: If True, also return the objectives matrix.
    
    Returns:
    - F: Total fitness values (R + D).
    - matrix_ret_risks: (optional) The [MDD, -mean_return] matrix.
    """
    matrix_ret_risks = precompute_objectives(population, returns_matrix)
    R = raw_fitness(population, matrix_ret_risks)
    D = calculate_density(matrix_ret_risks, k)
    F = R + D
    return (F, matrix_ret_risks) if return_matrix else F