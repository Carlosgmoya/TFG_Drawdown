import numpy as np

# This module contains functions to normalize the objectives of the portfolios.

def normalize_objectives(matrix_ret_risks):
    """
    Normalize the objectives in the matrix.
    
    Parameters:
    - matrix_ret_risks: A 2D array containing the returns and risks of the population.
    
    Returns:
    - norm: A 2D array containing the normalized returns and risks.
    """

    norm = np.zeros_like(matrix_ret_risks) # Normalized matrix
    for i in range(2):
        fmin, fmax = matrix_ret_risks[i].min(), matrix_ret_risks[i].max() # Minimum and maximum values for return and risk
        norm[i] = (matrix_ret_risks[i] - fmin) / (fmax - fmin + 1e-10) # Normalization
    return norm