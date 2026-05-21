import numpy as np

# This module contains functions to evaluate individuals and populations for:
# Objective 1: Maximum Drawdown (MDD) - to minimize
# Objective 2: Negative Mean Return - to minimize

def evaluate(individual, returns_matrix):
    """
    Evaluate the portfolio represented by the individual.
    """
    # Limpiar datos de entrada
    individual = np.nan_to_num(individual, nan=0.0, posinf=0.0, neginf=0.0)
    individual = np.clip(individual, 0, 1)  # Pesos entre 0 y 1
    
    returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    returns_matrix = np.clip(returns_matrix, -0.5, 0.5)  # Rendimientos diarios maximos +/-50%
    
    # 1. Portfolio returns
    portfolio_returns = np.dot(individual, returns_matrix)
    portfolio_returns = np.nan_to_num(portfolio_returns, nan=0.0)
    portfolio_returns = np.clip(portfolio_returns, -0.5, 0.5)
    
    # 2. Compounded wealth
    wealth = np.cumprod(1 + portfolio_returns)
    wealth = np.insert(wealth, 0, 1.0)
    wealth = np.clip(wealth, 0.001, 1000.0)  # Evitar extremos
    
    # 3. Running maximum
    running_max = np.maximum.accumulate(wealth)
    running_max = np.clip(running_max, 0.001, 1000.0)
    
    # 4. Drawdowns
    drawdowns = 1 - wealth / running_max
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)
    drawdowns = np.clip(drawdowns, 0, 1)
    
    # 5. Maximum Drawdown
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # 6. Mean return
    mean_return = np.mean(portfolio_returns)
    mean_return = np.clip(mean_return, -0.1, 0.1)  # Maximo +/-10% diario
    
    return mdd, -mean_return


def precompute_objectives(population, returns_matrix):
    """
    Precompute the MDD and negative mean returns for all individuals in the population.
    
    Parameters:
    - population: A 2D array of shape (pop_size, n_assets) representing the population.
    - returns_matrix: Historical returns matrix of shape (n_assets, q_periods).
    
    Returns:
    - matrix: A 2D array of shape (2, pop_size) with [MDD, -mean_return] for each individual.
    """
    N = len(population)
    matrix = np.zeros((2, N))
    for i in range(N):
        mdd, neg_ret = evaluate(population[i], returns_matrix)
        matrix[0, i] = mdd
        matrix[1, i] = neg_ret
    
    return matrix