import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .evaluation import precompute_objectives

# This module contains functions to visualize the Pareto front for MDD-based portfolios.

def plot_pareto_front(population_A, returns_matrix, cardinality, alg_name):
    """
    Plot the Pareto front of the population in MDD vs Mean Return space.
    
    Parameters:
    - population_A: The archive/final population of portfolios.
    - returns_matrix: Historical returns matrix (n_assets, n_periods).
    - cardinality: Maximum number of assets allowed.
    - alg_name: Name of the algorithm (for plot label).
    """
    
    # Precompute the objectives for the population
    # pareto_points[0, :] = MDD, pareto_points[1, :] = -mean_return
    pareto_points = precompute_objectives(population_A, returns_matrix)
    
    # Convert -mean_return back to positive mean return for visualization
    mdd_values = pareto_points[0, :] * 100  # Convert to percentage
    mean_returns = -pareto_points[1, :] * 100  # Negate and convert to percentage
    
    # ============================================
    # 1. Distribution of active assets
    # ============================================
    assets_list = np.zeros(cardinality + 1)
    
    for ind in population_A:
        # Count assets with weight > 0.1% (threshold to ignore negligible weights)
        indices = len(np.where(ind > 0.001)[0])
        if indices <= cardinality:
            assets_list[indices] += 1
        else:
            assets_list[cardinality] += 1  # Cap at cardinality
    
    df_dist = pd.DataFrame({
        'Num_Assets': np.arange(cardinality + 1),
        'Num_Solutions': assets_list.astype(int)
    })
    
    print(f"\n{'='*50}")
    print(f"Distribution of Active Assets - {alg_name}")
    print(f"{'='*50}")
    print(df_dist.to_string(index=False))
    
    # ============================================
    # 2. Pareto Front Plot: MDD vs Mean Return
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create DataFrame for plotting
    df_pareto = pd.DataFrame({
        'MDD (%)': mdd_values,
        'Mean Return (%)': mean_returns
    })
    
    # Plot Pareto front
    ax.scatter(
        df_pareto['MDD (%)'], 
        df_pareto['Mean Return (%)'],
        color='darkred', 
        alpha=0.7, 
        s=30,
        label=f'{alg_name} Pareto Front',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Labels and title
    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Pareto Front: MDD vs Mean Return - {alg_name}', fontsize=14, fontweight='bold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    # Add annotations for extreme points
    if len(mdd_values) > 0:
        # Point with minimum MDD
        min_mdd_idx = np.argmin(mdd_values)
        ax.annotate(
            f'Min MDD\n({mdd_values[min_mdd_idx]:.1f}%, {mean_returns[min_mdd_idx]:.2f}%)',
            xy=(mdd_values[min_mdd_idx], mean_returns[min_mdd_idx]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->')
        )
        
        # Point with maximum return
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
    plt.show()
    
    # ============================================
    # 3. Additional analysis: Drawdown Duration Distribution
    # ============================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # Calculate drawdown durations for the Pareto front
    durations = []
    for individual in population_A:
        portfolio_returns = np.dot(individual, returns_matrix)
        wealth = np.cumprod(1 + portfolio_returns)
        wealth = np.insert(wealth, 0, 1.0)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = 1 - wealth / running_max
        
        # Find longest drawdown period
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
    plt.show()