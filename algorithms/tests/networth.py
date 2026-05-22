"""
Validacion de Networth con 1000 EUR iniciales.
Uso: python networth_validation.py --prob 1
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from datetime import timedelta
from algorithms.utils.evaluation import evaluate


def get_problem_config(prob_num):
    """Devuelve la configuracion segun el numero de problema."""
    configs = {
        1: {
            'name': 'prob1_GrupoA_2025',
            'data_dir': 'data/prob1',
            'descripcion': 'Grupo A (10 grandes) - Train: 2025, Val: Ene 2026'
        },
        2: {
            'name': 'prob2_GrupoA_2024',
            'data_dir': 'data/prob2',
            'descripcion': 'Grupo A (10 grandes) - Train: 2024, Val: Ene 2025'
        },
        3: {
            'name': 'prob3_GrupoB_2025',
            'data_dir': 'data/prob3',
            'descripcion': 'Grupo B (10 diferentes) - Train: 2025, Val: Ene 2026'
        },
        4: {
            'name': 'prob4_GrupoB_2024',
            'data_dir': 'data/prob4',
            'descripcion': 'Grupo B (10 diferentes) - Train: 2024, Val: Ene 2025'
        },
    }
    
    if prob_num not in configs:
        print(f"[ERROR] Problema {prob_num} no valido. Usa 1, 2, 3 o 4.")
        sys.exit(1)
    
    return configs[prob_num]

def load_validation_data(csv_path):
    """Carga los datos de validacion."""
    val_returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns_matrix_val = val_returns.values.T
    return returns_matrix_val, val_returns


def load_all_populations(results_dir):
    """Carga todas las poblaciones finales guardadas."""
    pattern = f"{results_dir}/*_*/final_population.npy"
    pop_files = glob.glob(pattern)
    
    populations = {}
    for f in pop_files:
        folder_name = os.path.basename(os.path.dirname(f))
        algo_name = folder_name.split('_')[0]
        populations[algo_name] = np.load(f)
        print(f"  Cargado: {algo_name} ({len(populations[algo_name])} carteras)")
    
    return populations


def select_best_cartera(population, returns_matrix_train, criterio='min_mdd'):
    """Selecciona la mejor cartera segun el criterio."""
    from algorithms.utils.evaluation import precompute_objectives
    
    objectives = precompute_objectives(population, returns_matrix_train)
    
    if criterio == 'min_mdd':
        idx = np.argmin(objectives[0, :])
    elif criterio == 'max_return':
        idx = np.argmin(objectives[1, :])
    elif criterio == 'best_sortino':
        sortinos = []
        for i in range(len(population)):
            mdd, neg_ret = evaluate(population[i], returns_matrix_train)
            ret = -neg_ret
            sortinos.append(ret / (mdd + 1e-10))
        idx = np.argmax(sortinos)
    
    return population[idx]


def simulate_networth(weights, returns_matrix_val, initial_capital=1000):
    """Simula la evolucion del capital invertido."""
    portfolio_returns = np.dot(weights, returns_matrix_val)
    
    networth = initial_capital * np.cumprod(1 + portfolio_returns)
    networth = np.insert(networth, 0, initial_capital)
    
    running_max = np.maximum.accumulate(networth)
    drawdowns = (networth - running_max) / running_max * 100
    
    return networth, drawdowns


def plot_networth_comparison(all_results, val_dates, output_dir):
    """Graficos de evolucion del capital."""
    os.makedirs(output_dir, exist_ok=True)
    
    criterios = ['min_mdd', 'best_sortino', 'max_return']
    nombres = ['Min MDD', 'Best Sortino', 'Max Return']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for ax, criterio, nombre in zip(axes, criterios, nombres):
        for algo_name, data in all_results.items():
            networth = data[criterio]['networth']
            ax.plot(val_dates, networth, label=algo_name, linewidth=2, alpha=0.8)
        
        ax.axhline(y=1000, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Fecha', fontsize=11)
        ax.set_ylabel('Capital (EUR)', fontsize=11)
        ax.set_title(f'Cartera: {nombre}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/networth_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Grafico Min MDD
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    
    for algo_name, data in all_results.items():
        networth = data['min_mdd']['networth']
        ax2.plot(val_dates, networth, label=algo_name, linewidth=2.5, alpha=0.9)
    
    ax2.axhline(y=1000, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Capital inicial (1000 EUR)')
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Capital (EUR)', fontsize=12)
    ax2.set_title('Evolucion del Capital - Cartera de Minimo MDD', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/networth_min_mdd.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_networth_results(all_results, val_dates, output_dir):
    """Guarda resultados en archivo .txt"""
    
    lines = []
    lines.append("=" * 70)
    lines.append("  VALIDACION DE NETWORTH - 1000 EUR INICIALES")
    lines.append("=" * 70)
    lines.append(f"  Periodo: {val_dates[1].date()} a {val_dates[-1].date()}")
    lines.append(f"  Dias: {len(val_dates)-1}")
    
    criterios_nombres = [
        ('min_mdd', 'Minimo MDD'),
        ('best_sortino', 'Best Sortino'),
        ('max_return', 'Maximo Retorno')
    ]
    
    for criterio, nombre in criterios_nombres:
        lines.append(f"\n  CARTERA: {nombre}")
        lines.append(f"  {'Algoritmo':<12} {'Capital Final':>14} {'Rentabilidad':>12}")
        lines.append("  " + "-" * 40)
        
        ranking = sorted(all_results.items(), 
                        key=lambda x: x[1][criterio]['networth'][-1], 
                        reverse=True)
        
        for algo_name, data in ranking:
            capital_final = data[criterio]['networth'][-1]
            rentabilidad = (capital_final / 1000 - 1) * 100
            lines.append(f"  {algo_name:<12} {capital_final:>10.2f} EUR {rentabilidad:>10.2f}%")
    
    lines.append(f"\n  MEJOR CARTERA:")
    best = sorted(all_results.items(), 
                  key=lambda x: x[1]['min_mdd']['networth'][-1], 
                  reverse=True)[0]
    lines.append(f"  {best[0]}: {best[1]['min_mdd']['networth'][-1]:.2f} EUR")
    
    output_str = "\n".join(lines)
    
    with open(f'{output_dir}/networth_results.txt', 'w', encoding='utf-8') as f:
        f.write(output_str)
    
    print(output_str)


def main():
    parser = argparse.ArgumentParser(description='Validacion de Networth')
    parser.add_argument('--prob', type=int, required=True, 
                        help='Numero de problema (1-4)')
    args = parser.parse_args()
    
    config = get_problem_config(args.prob)
    
    print("=" * 70)
    print(f"  VALIDACION DE NETWORTH - {config['name']}")
    print("=" * 70)
    
    data_dir = config['data_dir']
    val_path = f'{data_dir}/val_returns.csv'
    train_path = f'{data_dir}/returns_matrix_train.npy'
    results_dir = f"results/{config['name']}"
    
    if not os.path.exists(val_path) or not os.path.exists(train_path):
        print(f"[ERROR] Datos no encontrados en {data_dir}")
        sys.exit(1)
    
    if not os.path.exists(results_dir):
        print(f"[ERROR] No se encuentra {results_dir}")
        print("Ejecuta primero main.py --prob {args.prob}")
        sys.exit(1)
    
    print("\nCargando datos...")
    returns_matrix_val, val_returns = load_validation_data(val_path)
    returns_matrix_train = np.load(train_path)
    
    val_dates = val_returns.index
    first_date = val_dates[0]
    
    print(f"  Validacion: {len(val_dates)} dias ({val_dates[0].date()} a {val_dates[-1].date()})")
    
    print("\nCargando poblaciones finales...")
    populations = load_all_populations(results_dir)
    
    if len(populations) == 0:
        print(f"\n[ERROR] No se encontraron poblaciones en {results_dir}")
        sys.exit(1)
    
    print("\nSimulando evolucion del capital...")
    
    all_results = {}
    criterios = ['min_mdd', 'best_sortino', 'max_return']
    
    for algo_name, population in populations.items():
        all_results[algo_name] = {}
        
        for criterio in criterios:
            best_weights = select_best_cartera(population, returns_matrix_train, criterio)
            networth, drawdowns = simulate_networth(best_weights, returns_matrix_val)
            
            all_results[algo_name][criterio] = {
                'networth': networth,
                'drawdowns': drawdowns,
                'weights': best_weights
            }
        
        capital_final = all_results[algo_name]['min_mdd']['networth'][-1]
        print(f"  {algo_name:<12} -> Min MDD: {capital_final:.2f} EUR")
    
    plot_dates = [first_date - timedelta(days=1)] + list(val_dates)
    
    output_dir = f"results/{config['name']}/networth"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerando graficos...")
    plot_networth_comparison(all_results, plot_dates, output_dir)
    
    save_networth_results(all_results, plot_dates, output_dir)
    
    print(f"\n[OK] Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()