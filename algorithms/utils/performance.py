import numpy as np
from algorithms.utils.evaluation import precompute_objectives, evaluate

def hypervolume(points):
    """Calculate hypervolume of points in MDD vs -Return space."""
    F = points.T.copy()
    
    F[:, 0] = 1.0 - (F[:, 0] - np.min(F[:, 0])) / (np.max(F[:, 0]) - np.min(F[:, 0]) + 1e-10)
    F[:, 1] = 1.0 - (F[:, 1] - np.min(F[:, 1])) / (np.max(F[:, 1]) - np.min(F[:, 1]) + 1e-10)
    F = F[np.argsort(F[:, 0])]
    
    ref_point = np.array([0.0, 0.0])
    hypervolume_val = 0.0
    prev_x = ref_point[0]
    
    for mdd_norm, ret_norm in F:
        width = mdd_norm - prev_x
        height = ret_norm - ref_point[1]
        hypervolume_val += width * height
        prev_x = mdd_norm
    
    return hypervolume_val


def sortino_ratio(population, returns_matrix, risk_free_rate=0.0):
    """Calculate average Sortino ratio."""
    sortino_ratios = np.zeros(len(population))
    
    for idx, individual in enumerate(population):
        portfolio_returns = np.dot(individual, returns_matrix)
        mean_return = np.mean(portfolio_returns)
        
        downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
            if downside_deviation > 0:
                sortino_ratios[idx] = (mean_return - risk_free_rate) / downside_deviation
    
    return np.mean(sortino_ratios) if len(population) > 0 else 0


def max_drawdown_metrics(population, returns_matrix):
    """Calculate MDD and drawdown duration."""
    mdds = np.zeros(len(population))
    durations = np.zeros(len(population))
    
    for idx, individual in enumerate(population):
        mdd, _ = evaluate(individual, returns_matrix)
        mdds[idx] = mdd
        
        portfolio_returns = np.dot(individual, returns_matrix)
        wealth = np.cumprod(1 + portfolio_returns)
        wealth = np.insert(wealth, 0, 1.0)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = 1 - wealth / running_max
        
        is_drawdown = drawdowns > 0
        if np.any(is_drawdown):
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


def calculate_performance(population, returns_matrix, save_path=None):
    """
    Calculate and optionally save performance metrics.
    
    Parameters:
    - population: Final population
    - returns_matrix: Historical returns matrix
    - save_path: If provided, save metrics to this file
    
    Returns:
    - metrics: Dictionary with all calculated metrics
    """
    
    objectives = precompute_objectives(population, returns_matrix)
    
    hypervol = hypervolume(objectives)
    sortino = sortino_ratio(population, returns_matrix)
    avg_mdd, avg_duration = max_drawdown_metrics(population, returns_matrix)
    
    output_lines = [
        f"Performance Metrics",
        f"{'='*40}",
        f"Hypervolume:              {hypervol:.4f}",
        f"Average Sortino Ratio:    {sortino:.4f}",
        f"Average MDD:              {avg_mdd:.4f} ({avg_mdd*100:.2f}%)",
        f"Average Max DD Duration:  {avg_duration:.1f} periods"
    ]
    
    output_str = "\n".join(output_lines)
    print(output_str)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(output_str)
        print(f"Métricas guardadas en: {save_path}")
    
    return {
        'hypervolume': hypervol,
        'sortino_ratio': sortino,
        'avg_mdd': avg_mdd,
        'avg_duration': avg_duration
    }