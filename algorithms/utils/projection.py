import numpy as np

# This module contains functions to normalize an individual and project it onto the simplex of probability.

def projection_simplex(y):
    """
    Projects a vector y onto the simplex defined by the constraints:
        Ω2 := {x ∈ ℝ^S : x ≥ 0, sum(x) = 1}
    """

    y_indexes = np.where(y > 0)[0] # Positive indexes of y
    y2 = y[y_indexes] # y2 is the vector of positive values of y
    S = len(y2) # Number of positive values in y
    y2 = np.asarray(y2) # Convert y2 to a numpy array
    u = np.sort(y2)[::-1] # Sort y2 in descending order
    cssv = np.cumsum(u) # Cumulative sum of u
    rho = np.max(np.nonzero(u + (1 - cssv) / (np.arange(1, S + 1)) > 0)[0]) # Find the maximum index where the condition is satisfied
    lambdaa = (1 - cssv[rho]) / (rho + 1) # Calculate the Lagrange multiplier
    z = np.maximum(y2 + lambdaa, 0) # Project y2 onto the simplex
    z2 = np.zeros(len(y)) # Create a new array of the same size as y
    z2[y_indexes] = z # Assign the projected values to the original array
    return z2