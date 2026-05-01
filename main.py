from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.data_loader import load_dataset, load_dataset_drawdown
from algorithms.utils.visualization import plot_pareto_front
from algorithms.utils.performance import calculate_performance
import numpy as np

if __name__ == "__main__":

    # Load dataset and run the algorithm, load .txt files preprocessed by the data_loader module
    returns = load_dataset_drawdown('data/real_data/raw_data/49_Industry_Portfolios_Daily.csv')
    cov_matrix = None  # Not needed for drawdown evaluation, but can be used for other objectives


    # Parameters for the algorithms
    # These parameters can be adjusted based on the specific requirements of the problem
    num_assets = returns.shape[1]  # Number of assets (N)
    N_arc = 250 # Archive population size (A0)
    N_pop = 250 # Usual population size (B0)
    generations = 400
    cardinality = 10
    crossover_rate = 0.9
    mutation_rate = 1 / num_assets

    # NPGA2 specific parameters
    tournament_size = 10
    niche_radius = 0.7

    # PESA specific parameters
    grid_divisions = 10

    # E-MOEA specific parameters
    e = 0.00458 * 6


    # Uncomment the algorithm you want to run
 
    # Run NSGA-II 
    """nsga2 = NSGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = nsga2.evolve()
    calculate_performance(final_population, returns, cov_matrix)
    plot_pareto_front(final_population, returns, cov_matrix, cardinality, "NSGA-II")"""


    # Run SPEA-II algorithm
    """spea2 = SPEA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = spea2.evolve()
    calculate_performance(final_population, returns, cov_matrix)
    plot_pareto_front(final_population, returns, cov_matrix, cardinality, "SPEA2")"""


    # Run NPGA2 algorithm    
    """npga2 = NPGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations, tournament_size, niche_radius)
    final_population = npga2.evolve()
    calculate_performance(final_population, returns, cov_matrix)
    plot_pareto_front(final_population, returns, cov_matrix, cardinality, "NPGA2")"""


    # Run PESA algorithm    
    """pesa = PESA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, grid_divisions)
    final_population = pesa.evolve()
    calculate_performance(final_population, returns, cov_matrix)
    plot_pareto_front(final_population, returns, cov_matrix, cardinality, "PESA")"""

    # Run E-MOEA algorithm
    """e_moea = E_MOEA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, e)
    final_population = e_moea.evolve()
    calculate_performance(final_population, returns, cov_matrix)
    plot_pareto_front(final_population, returns, cov_matrix, cardinality, "e-MOEA")"""
            
    print("Shape:", returns.shape)
    print("NaNs:", np.isnan(returns).sum())
    print("Infs:", np.isinf(returns).sum())
    print("Min return:", np.min(returns))
    print("Max return:", np.max(returns))