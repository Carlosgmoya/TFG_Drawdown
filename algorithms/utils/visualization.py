import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .evaluation import precompute_objectives

# This module contains functions to visualize the Pareto front of the population.

def plot_pareto_front(population_A, returns, cov_matrix, cardinality, alg_name):
        """
        Plot the Pareto front of the population.
        
        """

        # Precompute the objectives for the archive population
        pareto_points = precompute_objectives(population_A, returns, cov_matrix)

        # Calculate the number of active assets in each solution
        # and create a DataFrame to display the distribution
        assets_list = np.zeros(cardinality + 1)

        for ind in population_A:
            indices = len(np.where(ind > 0)[0])
            assets_list[indices] += 1

        df_dist = pd.DataFrame({
            'Num_Assets': np.arange(cardinality + 1),
            'Num_Solutions': assets_list.astype(int)
        })

        # Display the distribution of active assets
        print(df_dist.to_string(index=False))

        # Plot the distribution of active assets
        df_pareto = pd.DataFrame({
            'Mean': pareto_points[0, :],
            'Variance': pareto_points[1, :]
        })

        # Plot the Pareto front
        ax = df_pareto.plot.scatter(x='Variance', y='Mean', color='red', label=alg_name, figsize=(8, 6))
        ax.set_xlabel('Variance', fontweight='bold')
        ax.set_ylabel('Mean', fontweight='bold')
        ax.set_title('Portfolio Optimization', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True)
        plt.show()