"""
Main script para optimizacion de carteras con MDD.
Uso: python main.py --prob 1
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.visualization import plot_pareto_front
from algorithms.utils.performance import calculate_performance


def get_problem_config(prob_num):
    """
    Devuelve la configuracion segun el numero de problema.
    """
    configs = {
        1: {
            'name': 'prob1_GrupoA_2025',
            'data_dir': 'data/prob1',
            'descripcion': 'Grupo A - 2025 (Train: 2025, Val: Ene 2026)'
        },
        2: {
            'name': 'prob2_GrupoA_2024',
            'data_dir': 'data/prob2',
            'descripcion': 'Grupo A - 2024 (Train: 2024, Val: Ene 2025)'
        },
        3: {
            'name': 'prob3_GrupoB_2025',
            'data_dir': 'data/prob3',
            'descripcion': 'Grupo B - 2025 (Train: 2025, Val: Ene 2026)'
        },
        4: {
            'name': 'prob4_GrupoB_2024',
            'data_dir': 'data/prob4',
            'descripcion': 'Grupo B - 2024 (Train: 2024, Val: Ene 2025)'
        },
    }
    
    if prob_num not in configs:
        print(f"[ERROR] Problema {prob_num} no valido. Usa 1, 2, 3 o 4.")
        sys.exit(1)
    
    return configs[prob_num]


if __name__ == "__main__":

    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Optimizacion de carteras con MDD')
    parser.add_argument('--prob', type=int, required=True, 
                        help='Numero de problema (1-4)')
    args = parser.parse_args()
    
    # Obtener configuracion
    config = get_problem_config(args.prob)
    
    print("=" * 70)
    print(f"  OPTIMIZACION MDD - {config['descripcion']}")
    print("=" * 70)
    
    # ============================================
    # CARGAR DATOS
    # ============================================
    data_path = f"{config['data_dir']}/returns_matrix_train.npy"
    
    if not os.path.exists(data_path):
        print(f"[ERROR] No se encuentra {data_path}")
        print("Ejecuta primero prepare_ibex_data.py para generar los datasets")
        sys.exit(1)
    
    print(f"\nCargando datos desde {data_path}...")
    returns_matrix = np.load(data_path)
    
    num_assets = returns_matrix.shape[0]
    num_periods = returns_matrix.shape[1]
    print(f"Datos cargados: {num_assets} activos, {num_periods} dias")

    # ============================================
    # PARAMETROS GENERALES
    # ============================================
    N_arc = 250
    N_pop = 250
    generations = 400
    cardinality = num_assets
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
    base_results_dir = f"results/{config['name']}"
    os.makedirs(base_results_dir, exist_ok=True)

    # ============================================
    # EJECUTAR ALGORITMOS (descomenta el que quieras probar)
    # ============================================

    # Run NSGA-II 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/NSGA-II_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    nsga2 = NSGA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = nsga2.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics.txt")
    np.save(f"{results_dir}/final_population.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NSGA-II", save_path=f"{results_dir}/pareto.png")
    print(f"\\n[OK] Resultados NSGA-II guardados en: {results_dir}")
    

    # Run SPEA-II algorithm
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/SPEA2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    spea2 = SPEA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = spea2.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics.txt")
    np.save(f"{results_dir}/final_population.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "SPEA2", save_path=f"{results_dir}/pareto.png")
    print(f"\\n[OK] Resultados SPEA2 guardados en: {results_dir}")
    

    # Run NPGA2 algorithm    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/NPGA2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    npga2 = NPGA2(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate, mutation_rate, generations, tournament_size, niche_radius)
    final_population = npga2.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics.txt")
    np.save(f"{results_dir}/final_population.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "NPGA2", save_path=f"{results_dir}/pareto.png")
    print(f"\\n[OK] Resultados NPGA2 guardados en: {results_dir}")
    

    # Run PESA algorithm    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/PESA_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    pesa = PESA(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, grid_divisions)
    final_population = pesa.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics.txt")
    np.save(f"{results_dir}/final_population.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "PESA", save_path=f"{results_dir}/pareto.png")
    print(f"\\n[OK] Resultados PESA guardados en: {results_dir}")
    

    # Run E-MOEA algorithm
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_results_dir}/e-MOEA_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    e_moea = E_MOEA(N_arc, N_pop, num_assets, returns_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, e)
    final_population = e_moea.evolve()
    calculate_performance(final_population, returns_matrix, save_path=f"{results_dir}/metrics.txt")
    np.save(f"{results_dir}/final_population.npy", final_population)
    plot_pareto_front(final_population, returns_matrix, cardinality, "e-MOEA", save_path=f"{results_dir}/pareto.png")
    print(f"\\n[OK] Resultados e-MOEA guardados en: {results_dir}")
    