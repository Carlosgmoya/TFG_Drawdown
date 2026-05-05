import numpy as np
from algorithms.utils.evaluation import precompute_objectives, evaluate

# This module contains functions to calculate performance metrics for MDD-based portfolios.

def hypervolume(points):
    """
    Calculate the hypervolume of the given points in MDD vs -Return space.
    
    Parameters:
    - points: A 2D array of shape (2, N) representing [MDD, -mean_return] for each individual.
    
    Returns:
    - hypervolume: The hypervolume of the points (higher is better).
    """
    F = points.T.copy()  # Shape (N, 2)
    
    # Normalize to [0, 1] range
    # MDD (column 0): 0 is best (no drawdown), 1 is worst. We invert so 1 = best.
    F[:, 0] = 1.0 - (F[:, 0] - np.min(F[:, 0])) / (np.max(F[:, 0]) - np.min(F[:, 0]) + 1e-10)
    
    # -Mean return (column 1): more negative = worse. We invert so 1 = best.
    F[:, 1] = 1.0 - (F[:, 1] - np.min(F[:, 1])) / (np.max(F[:, 1]) - np.min(F[:, 1]) + 1e-10)
    
    # Sort by first objective (MDD normalized)
    F = F[np.argsort(F[:, 0])]
    
    # Reference point (worst possible: 0, 0 after normalization)
    ref_point = np.array([0.0, 0.0])
    
    hypervolume = 0.0
    prev_x = ref_point[0]
    
    # Calculate hypervolume (area dominated by the Pareto front)
    for mdd_norm, ret_norm in F:
        width = mdd_norm - prev_x
        height = ret_norm - ref_point[1]
        hypervolume += width * height
        prev_x = mdd_norm
    
    return hypervolume


def sortino_ratio(population, returns_matrix, risk_free_rate=0.0):
    """
    Calculate the Sortino ratio for each individual in the population.
    Sortino ratio uses downside deviation instead of total standard deviation.
    
    Parameters:
    - population: Array of portfolios (weights).
    - returns_matrix: Historical returns matrix (n_assets, n_periods).
    - risk_free_rate: Risk-free rate (default 0).
    
    Returns:
    - avg_sortino: Average Sortino ratio of the population.
    """
    sortino_ratios = np.zeros(len(population))
    
    for idx, individual in enumerate(population):
        # Calculate portfolio returns over time
        portfolio_returns = np.dot(individual, returns_matrix)
        
        # Mean return
        mean_return = np.mean(portfolio_returns)
        
        # Downside deviation (only consider returns below risk-free rate)
        downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
            if downside_deviation > 0:
                sortino_ratios[idx] = (mean_return - risk_free_rate) / downside_deviation
        
    return np.mean(sortino_ratios) if len(population) > 0 else 0


def max_drawdown_metrics(population, returns_matrix):
    """
    Calculate MDD and drawdown duration for each individual.
    
    Parameters:
    - population: Array of portfolios.
    - returns_matrix: Historical returns matrix.
    
    Returns:
    - avg_mdd: Average Maximum Drawdown.
    - avg_duration: Average drawdown duration (periods underwater).
    """
    mdds = np.zeros(len(population))
    durations = np.zeros(len(population))
    
    for idx, individual in enumerate(population):
        mdd, _ = evaluate(individual, returns_matrix)
        mdds[idx] = mdd
        
        # Calculate drawdown duration
        portfolio_returns = np.dot(individual, returns_matrix)
        wealth = np.cumprod(1 + portfolio_returns)
        wealth = np.insert(wealth, 0, 1.0)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = 1 - wealth / running_max
        
        # Duration: count consecutive periods where drawdown > 0
        is_drawdown = drawdowns > 0
        if np.any(is_drawdown):
            # Find longest drawdown period
            max_duration = 0
            current_duration = 0
            for dd in is_drawdown:
                if dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            durations[idx] = max_duration
    
    return np.mean(mdds), np.mean(durations)


def calculate_performance(population, returns_matrix):
    """
    Calculate performance metrics for MDD-based portfolio optimization.
    
    Parameters:
    - population: The final population from the evolutionary algorithm.
    - returns_matrix: Historical returns matrix (n_assets, n_periods).
    
    Returns:
    - performance: Dictionary with hypervolume, Sortino ratio, MDD, and drawdown duration.
    """
    
    # Get objectives for all individuals
    objectives = precompute_objectives(population, returns_matrix)
    # objectives[0, :] = MDD, objectives[1, :] = -mean_return
    
    # 1. Hypervolume
    hypervol = hypervolume(objectives)
    
    # 2. Sortino ratio (using drawdown-aware risk metric)
    sortino = sortino_ratio(population, returns_matrix)
    
    # 3. Average MDD and drawdown duration
    avg_mdd, avg_duration = max_drawdown_metrics(population, returns_matrix)
    
    print(f"Performance Metrics:")
    print(f"  Hypervolume: {hypervol:.4f}")
    print(f"  Average Sortino Ratio: {sortino:.4f}")
    print(f"  Average MDD: {avg_mdd:.4f} ({avg_mdd*100:.2f}%)")
    print(f"  Average Max Drawdown Duration: {avg_duration:.1f} periods")
    
    return {
        'hypervolume': hypervol,
        'sortino_ratio': sortino,
        'avg_mdd': avg_mdd,
        'avg_duration': avg_duration
    }