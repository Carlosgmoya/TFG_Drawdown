from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.data_loader import load_historical_returns
from algorithms.utils.visualization import plot_pareto_front
from algorithms.utils.performance import calculate_performance

if __name__ == "__main__":

    # ============================================
    # CHANGE: Load historical returns matrix instead of returns and cov_matrix
    # ============================================
    # Option 1: If you have a CSV file with historical prices/returns
    returns_matrix, asset_names = load_historical_returns('data/real_data/your_historical_data.csv')
    
    # Option 2: If your data is already in the right format
    # returns_matrix = np.loadtxt('data/synthetic_data/Japanese_Nikkei_225.txt').T
    
    num_assets = returns_matrix.shape[0]
    num_periods = returns_matrix.shape[1]
    
    print(f"Loaded data: {num_assets} assets, {num_periods} periods")

    # ============================================
    # Parameters for the algorithms
    # ============================================
    N_arc = 250          # Archive population size (A0)
    N_pop = 250          # Usual population size (B0)
    generations = 400
    cardinality = 10     # For your TFG, consider if you want to keep this constraint
    crossover_rate = 0.9
    mutation_rate = 1 / num_assets

    # NPGA2 specific parameters
    tournament_size = 10
    niche_radius = 0.7

    # PESA specific parameters
    grid_divisions = 10

    # E-MOEA specific parameters
    e = 0.00458 * 6


    # ============================================
    # Run all algorithms - uncomment the one you want to test
    # ============================================

    # Run NSGA-II 
    nsga2 = NSGA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = nsga2.evolve()
    calculate_performance(final_population, returns_matrix)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NSGA-II_MDD")


    # Run SPEA-II algorithm
    """
    spea2 = SPEA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = spea2.evolve()
    calculate_performance(final_population, returns_matrix)
    plot_pareto_front(final_population, returns_matrix, cardinality, "SPEA2_MDD")
    """


    # Run NPGA2 algorithm    
    """
    npga2 = NPGA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations, tournament_size, niche_radius)
    final_population = npga2.evolve()
    calculate_performance(final_population, returns_matrix)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NPGA2_MDD")
    """


    # Run PESA algorithm    
    """
    pesa = PESA(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, grid_divisions)
    final_population = pesa.evolve()
    calculate_performance(final_population, returns_matrix)
    plot_pareto_front(final_population, returns_matrix, cardinality, "PESA_MDD")
    """


    # Run E-MOEA algorithm
    """
    e_moea = E_MOEA(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, e)
    final_population = e_moea.evolve()
    calculate_performance(final_population, returns_matrix)
    plot_pareto_front(final_population, returns_matrix, cardinality, "e-MOEA_MDD")
    """


    # ============================================
    # OPTIONAL: Comparison with classic Markowitz
    # ============================================
    """ 
    # If you want to compare with Markowitz approach:
    from algorithms.utils.data_loader import load_data_for_both_models
    
    data = load_data_for_both_models('data/real_data/your_historical_data.csv')
    
    # Run NSGA-II with MDD objectives
    nsga2_mdd = NSGA2(N_arc, N_pop, num_assets, data['returns_matrix'], 
                      cardinality, crossover_rate, mutation_rate, generations)
    pop_mdd = nsga2_mdd.evolve()
    
    # Run NSGA-II with Markowitz objectives (for comparison)
    nsga2_mv = NSGA2(N_arc, N_pop, num_assets, data['mean_returns'], 
                     data['cov_matrix'], cardinality, crossover_rate, mutation_rate, generations)
    pop_mv = nsga2_mv.evolve()
    
    # Compare frontiers
    plot_pareto_front(pop_mdd, data['returns_matrix'], cardinality, "NSGA-II_MDD")
    plot_pareto_front(pop_mv, data['mean_returns'], data['cov_matrix'], cardinality, "NSGA-II_Markowitz")
    """