import numpy as np
from algorithms.utils.evaluation import precompute_objectives

# This module contains functions to calculate performance metrics: hypervolume and Sharpe ratio.

def hypervolume(points):
    """
    Calculate the hypervolume of the given points.
    Parameters:
    - points: A 2D array representing the points in the objective space.
    Returns:
    - hypervolume: The hypervolume of the points.
    """

    F = points.T

    # Normalize the points to the range [0, 1]
    F[:, 0] = (F[:, 0] - np.min(F[:, 0])) / (np.max(F[:, 0]) - np.min(F[:, 0]))  # retun
    F[:, 1] = (F[:, 1] - np.min(F[:, 1])) / (np.max(F[:, 1]) - np.min(F[:, 1]))  # risk

    # Sort the points by the first objective (return)
    F = F[np.argsort(F[:, 0])]

    # Reference point for hypervolume calculation
    ref_point = np.array([0.0, 1.0])

    hypervolume = 0.0
    prev_y = ref_point[0]

    # Calculate the hypervolume
    for ret, var in F:
        width = ref_point[1] - var
        height = ret - prev_y
        hypervolume += width * height
        prev_y = ret

    return hypervolume


def sharpe_ratio(population, returns, cov_matrix):
    """
    Calculate the Sharpe ratio for the given population.
    Parameters:
    - population: The current population.
    Returns:
    - sharpe: The average Sharpe ratio of the population.
    """

    matrix_ret_risks = precompute_objectives(population, returns, cov_matrix)
    sharpe = np.zeros(len(population))

    # Calculate the Sharpe ratio for each individual in the population
    for idx, ind in enumerate(matrix_ret_risks.T):
        sharpe[idx] = ind[0] / ind[1] if ind[1] != 0 else 0

    # Return the average Sharpe ratio
    return sum(sharpe) / len(sharpe) if len(sharpe) > 0 else 0


def calculate_performance(population, returns, cov_matrix):
    """
    Calculate the performance metrics for the given population.
    Parameters:
    - population: The current population.
    Returns:
    - performance: A dictionary containing the hypervolume and Sharpe ratio.
    """

    hypervol = hypervolume(precompute_objectives(population, returns, cov_matrix))
    sharpe = sharpe_ratio(population, returns, cov_matrix)

    print(f"Hypervolume: {hypervol:.3f}, Sharpe Ratio: {sharpe:.3f}")