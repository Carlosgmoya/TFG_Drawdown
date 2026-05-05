import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .evaluation import precompute_objectives

# This module contains functions to visualize the Pareto front for MDD-based portfolios.

def plot_pareto_front(population_A, returns_matrix, cardinality, alg_name, save_path=None):
    """
    Plot the Pareto front of the population in MDD vs Mean Return space.
    
    Parameters:
    - population_A: The archive/final population of portfolios.
    - returns_matrix: Historical returns matrix (n_assets, n_periods).
    - cardinality: Maximum number of assets allowed.
    - alg_name: Name of the algorithm (for plot label).
    - save_path: If provided, save the figure to this path instead of showing it.
    """
    
    # Precompute objectives
    pareto_points = precompute_objectives(population_A, returns_matrix)
    
    # Convert to percentage for visualization
    mdd_values = pareto_points[0, :] * 100
    mean_returns = -pareto_points[1, :] * 100
    
    # Distribution of active assets
    assets_list = np.zeros(cardinality + 1)
    for ind in population_A:
        indices = len(np.where(ind > 0.001)[0])
        if indices <= cardinality:
            assets_list[indices] += 1
        else:
            assets_list[cardinality] += 1
    
    df_dist = pd.DataFrame({
        'Num_Assets': np.arange(cardinality + 1),
        'Num_Solutions': assets_list.astype(int)
    })
    
    print(f"\n{'='*50}")
    print(f"Distribution of Active Assets - {alg_name}")
    print(f"{'='*50}")
    print(df_dist.to_string(index=False))
    
    # Pareto Front Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.scatter(
        mdd_values, 
        mean_returns,
        color='darkred', 
        alpha=0.7, 
        s=30,
        label=f'{alg_name} Pareto Front',
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Pareto Front: MDD vs Mean Return - {alg_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    # Annotate extreme points
    if len(mdd_values) > 0:
        min_mdd_idx = np.argmin(mdd_values)
        ax.annotate(
            f'Min MDD\n({mdd_values[min_mdd_idx]:.1f}%, {mean_returns[min_mdd_idx]:.2f}%)',
            xy=(mdd_values[min_mdd_idx], mean_returns[min_mdd_idx]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->')
        )
        
        max_ret_idx = np.argmax(mean_returns)
        ax.annotate(
            f'Max Return\n({mdd_values[max_ret_idx]:.1f}%, {mean_returns[max_ret_idx]:.2f}%)',
            xy=(mdd_values[max_ret_idx], mean_returns[max_ret_idx]),
            xytext=(10, -10), textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->')
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
        plt.close()
    else:
        plt.show()
    
    # Drawdown Duration Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    durations = []
    for individual in population_A:
        portfolio_returns = np.dot(individual, returns_matrix)
        wealth = np.cumprod(1 + portfolio_returns)
        wealth = np.insert(wealth, 0, 1.0)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = 1 - wealth / running_max
        
        is_drawdown = drawdowns > 0
        max_duration = 0
        current_duration = 0
        for dd in is_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        durations.append(max_duration)
    
    ax2.hist(durations, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Maximum Drawdown Duration (periods)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution of Max Drawdown Duration - {alg_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        duration_path = save_path.replace('.png', '_duration.png')
        plt.savefig(duration_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico de duración guardado en: {duration_path}")
        plt.close()
    else:
        plt.show()