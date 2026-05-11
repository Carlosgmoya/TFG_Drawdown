import os
import numpy as np
from datetime import datetime

from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.visualization import plot_pareto_front
from algorithms.utils.performance import calculate_performance

if __name__ == "__main__":

    # ============================================
    # CARGAR DATOS
    # ============================================
    print("Cargando datos del IBEX 35...")
    returns_matrix = np.load('data/ibex35/returns_matrix_train.npy')
    
    num_assets = returns_matrix.shape[0]
    num_periods = returns_matrix.shape[1]
    print(f"Datos cargados: {num_assets} activos, {num_periods} días")

    # ============================================
    # PARÁMETROS GENERALES
    # ============================================
    N_arc = 250          # Archive population size (A0)
    N_pop = 250          # Usual population size (B0)
    generations = 400
    cardinality = num_assets  # Todos los activos disponibles
    crossover_rate = 0.9
    mutation_rate = 1 / num_assets

    # NPGA2 specific parameters
    tournament_size = 10
    niche_radius = 0.7

    # PESA specific parameters
    grid_divisions = 10

    # E-MOEA specific parameters
    e = 0.00458 * 6

    # Directorio base de resultados
    base_results_dir = "results"

    # ============================================
    # EJECUTAR ALGORITMOS (descomenta el que quieras probar)
    # ============================================

    # Run NSGA-II 
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/NSGA-II_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    nsga2 = NSGA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = nsga2.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics_NSGA-II.txt")
    np.save(f"{results_dir}/final_population_NSGA-II.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NSGA-II", save_path=f"{results_dir}/pareto_NSGA-II.png")
    print(f"\n✅ Resultados NSGA-II guardados en: {results_dir}")
    """

    # Run SPEA-II algorithm
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/SPEA2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    spea2 = SPEA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = spea2.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics_SPEA2.txt")
    np.save(f"{results_dir}/final_population_SPEA2.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "SPEA2", save_path=f"{results_dir}/pareto_SPEA2.png")
    print(f"\n✅ Resultados SPEA2 guardados en: {results_dir}")
    """

    # Run NPGA2 algorithm    
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/NPGA2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    npga2 = NPGA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations, tournament_size, niche_radius)
    final_population = npga2.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics_NPGA2.txt")
    np.save(f"{results_dir}/final_population_NPGA2.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NPGA2", save_path=f"{results_dir}/pareto_NPGA2.png")
    print(f"\n✅ Resultados NPGA2 guardados en: {results_dir}")
    """

    # Run PESA algorithm    
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/PESA_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    pesa = PESA(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, grid_divisions)
    final_population = pesa.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics_PESA.txt")
    np.save(f"{results_dir}/final_population_PESA.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "PESA", save_path=f"{results_dir}/pareto_PESA.png")
    print(f"\n✅ Resultados PESA guardados en: {results_dir}")
    """

    # Run E-MOEA algorithm
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/e-MOEA_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    e_moea = E_MOEA(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, e)
    final_population = e_moea.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics_e-MOEA.txt")
    np.save(f"{results_dir}/final_population_e-MOEA.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "e-MOEA", save_path=f"{results_dir}/pareto_e-MOEA.png")
    print(f"\n✅ Resultados e-MOEA guardados en: {results_dir}")
    