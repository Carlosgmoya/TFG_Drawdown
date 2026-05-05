from algorithms.nsga2 import NSGA2
from algorithms.utils.visualization import plot_pareto_front
from algorithms.utils.performance import calculate_performance
import numpy as np

if __name__ == "__main__":

    # ============================================
    # Cargar datos del IBEX 35 preparados
    # ============================================
    print("Cargando datos del IBEX 35...")
    returns_matrix = np.load('data/ibex35/returns_matrix_train.npy')
    
    num_assets = returns_matrix.shape[0]
    num_periods = returns_matrix.shape[1]
    
    print(f"Datos cargados: {num_assets} activos, {num_periods} días")

    # ============================================
    # Parámetros del algoritmo
    # ============================================
    N_arc = 100          # Reducido para prueba rápida
    N_pop = 100          # Reducido para prueba rápida
    generations = 50     # Reducido para prueba rápida
    cardinality = num_assets  # Permitir todos los activos
    crossover_rate = 0.9
    mutation_rate = 1 / num_assets

    # ============================================
    # Ejecutar NSGA-II con objetivos MDD
    # ============================================
    print("\nEjecutando NSGA-II con optimización MDD...")
    
    nsga2 = NSGA2(
        N_arc, N_pop, num_assets, returns_matrix,
        cardinality, crossover_rate, mutation_rate, generations
    )
    
    final_population = nsga2.evolve()
    
    # ============================================
    # Resultados
    # ============================================
    calculate_performance(final_population, returns_matrix)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NSGA-II_MDD_IBEX35")
    
    print("\n✅ Prueba completada!")