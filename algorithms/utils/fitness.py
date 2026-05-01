import numpy as np
from scipy.spatial.distance import cdist
from .evaluation import precompute_objectives

# This module contains functions to calculate the fitness of a population.

def dominates(matrix_ret_risks, ind1, ind2):
    """
    Check if individual 1 dominates individual 2.
    
    Parameters:
    - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        - ind1: First individual (portfolio).
        - ind2: Second individual (portfolio).

    Returns:
    - True if ind1 dominates ind2, False otherwise.
    """
    return1 = matrix_ret_risks[0][ind1]
    risk1 = matrix_ret_risks[1][ind1]
    return2 = matrix_ret_risks[0][ind2]
    risk2 = matrix_ret_risks[1][ind2]       
    return (return1 >= return2 and risk1 <= risk2) and (return1 > return2 or risk1 < risk2)


def raw_fitness(population, matrix_ret_risks):
    """
    Calculate the raw fitness values for each individual in the population.
    Parameters:
    - population: A 2D array representing the population of portfolios.
    - matrix_ret_risks: A 2D array containing the returns and risks of the population.
    Returns:
    - raw_fitness_values: A 1D array representing the raw fitness values of each individual.
    """
    N = len(population)
    dominance_count = np.zeros(N) # A 2D array that count of how many individuals dominate individual i
    dominated_sets = [[] for _ in range(N)] # List of dominated sets for each individual
    
    for i in range(N): # Index and portfolio
        for j in range(N): # Compare with all other portfolios
            if dominates(matrix_ret_risks, i, j): # If i dominates j
                dominated_sets[i].append(j)
            elif dominates(matrix_ret_risks, j, i): # If j dominates i
                dominance_count[i] += 1

    raw_fitness_values = np.zeros(N)
    for i in range(N):
        raw_fitness_values[i] = dominance_count[i] / (len(dominated_sets[i]) if len(dominated_sets[i]) > 0 else 1)

    return raw_fitness_values


def calculate_density(matrix_ret_risks, k=1):
    """
    Calculate the density of each individual in the population.
    
    Parameters:
    - matrix_ret_risks: A 2D array containing the returns and risks of the population.
    - k: Number of neighbors to consider.
    
    Returns:
    - D: A 1D array representing the density of each individual in the population.
    """
    points = matrix_ret_risks.T  # Shape (N, 2) 
    distances = cdist(points, points) # Compute distance between all points
    np.fill_diagonal(distances, np.inf) # Set diagonal to infinity to avoid self-comparison
    kth_distances = np.partition(distances, k, axis=1)[:, k] # k-th smallest distance for each point
    return 1.0 / (kth_distances + 2.0) # Density is the inverse of the k-th smallest distance


def calculate_total_fitness(population, returns, cov_matrix, k=1, return_matrix=False):
    """Calculate the total fitness of each individual in the population.
    
    Parameters:
    - population: A 2D array representing the population of portfolios.
    - returns: A 1D array representing the expected returns of each asset.
    - cov_matrix: A 2D array representing the covariance matrix of the assets.
    - k: Number of neighbors to consider.
    
    Returns:
    - F: A 1D array representing the total fitness of each individual in the population.
    - matrix_ret_risks: A 2D array containing the returns and risks of the population.
    """
    matrix_ret_risks = precompute_objectives(population, returns, cov_matrix)
    R = raw_fitness(population, matrix_ret_risks)
    D = calculate_density(matrix_ret_risks, k)
    F = R + D
    return (F, matrix_ret_risks) if return_matrix else F