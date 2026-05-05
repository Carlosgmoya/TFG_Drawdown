import numpy as np

# This module contains functions to evaluate individuals and populations for:
# Objective 1: Maximum Drawdown (MDD) - to minimize
# Objective 2: Negative Mean Return - to minimize

def evaluate(individual, returns_matrix):
    """
    Evaluate the portfolio represented by the individual.
    
    Parameters:
    - individual: A portfolio represented as a vector of weights (n_assets,).
    - returns_matrix: Historical returns matrix of shape (n_assets, q_periods).
                      Element [i, t] is the return of asset i in period t.
    
    Returns:
    - mdd: Maximum Drawdown of the portfolio (to minimize).
    - neg_mean_return: Negative mean return of the portfolio (to minimize).
    """
    # 1. Calculate portfolio returns over time: r_{p,t} = w^T * r_t for each t
    # returns_matrix has shape (n_assets, q_periods)
    # individual has shape (n_assets,)
    # Result: portfolio_returns of shape (q_periods,)
    portfolio_returns = np.dot(individual, returns_matrix)
    
    # 2. Calculate compounded wealth: W_t = ∏_{s=1}^t (1 + r_{p,s})
    # Start with wealth = 1 at t=0
    wealth = np.cumprod(1 + portfolio_returns)
    wealth = np.insert(wealth, 0, 1.0)  # W_0 = 1
    
    # 3. Calculate running maximum: W_max_t = max_{0≤s≤t} W_s
    running_max = np.maximum.accumulate(wealth)
    
    # 4. Calculate drawdown at each point: δ_t = 1 - W_t / W_max_t
    drawdowns = 1 - wealth / running_max
    
    # 5. Maximum Drawdown: MDD = max δ_t
    mdd = np.max(drawdowns)
    
    # 6. Mean return of the portfolio (sample mean over periods)
    mean_return = np.mean(portfolio_returns)
    
    # Return both objectives (minimizing MDD and maximizing return = minimizing -return)
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