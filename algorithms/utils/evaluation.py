import numpy as np

# =========================
# AUX FUNCTIONS
# =========================

def portfolio_returns(weights, returns_matrix):
    # returns_matrix: (T x N)
    return np.dot(returns_matrix, weights)


def compute_wealth(returns):
    return np.cumprod(1 + returns)


def compute_drawdown(wealth):
    peak = np.maximum.accumulate(wealth)
    return 1 - wealth / peak


def max_drawdown(returns):
    wealth = compute_wealth(returns)
    drawdown = compute_drawdown(wealth)
    return np.max(drawdown)


# =========================
# MAIN EVALUATION
# =========================

def evaluate(individual, returns_matrix, cov_matrix=None):
    """
    Evaluate portfolio using:
    - Maximum Drawdown (MDD)
    - Mean return
    
    Returns:
    - mdd (to minimize)
    - -mean_return (to minimize)
    """

    # 1. Portfolio returns time series
    port_returns = portfolio_returns(individual, returns_matrix)

    # 2. Mean return
    mean_return = np.mean(port_returns)

    # 3. Max Drawdown
    mdd = max_drawdown(port_returns)

    return mdd, -mean_return


def precompute_objectives(population, returns_matrix, cov_matrix=None):
    """
    Precompute objectives for the population:
    row 0 -> MDD
    row 1 -> -return
    """
    N = len(population)
    matrix = np.zeros((2, N))

    for i in range(N):
        matrix[:, i] = evaluate(population[i], returns_matrix)

    return matrix