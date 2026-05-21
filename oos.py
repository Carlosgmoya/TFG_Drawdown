"""
Validacion Out-of-Sample para las carteras optimizadas con MDD.
Uso: python oos.py --prob 1
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
    """Carga los datos de validacion out-of-sample."""
    val_returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns_matrix_val = val_returns.values.T
    
    print(f"Datos de validacion: {returns_matrix_val.shape[0]} activos, "
          f"{returns_matrix_val.shape[1]} dias")
    print(f"Periodo: {val_returns.index[0].date()} a {val_returns.index[-1].date()}")
    
    return returns_matrix_val, val_returns


def select_pareto_front(population, returns_matrix_train):
    """Selecciona las carteras no dominadas (frontera de Pareto)."""
    from algorithms.utils.evaluation import precompute_objectives
    
    objectives = precompute_objectives(population, returns_matrix_train)
    
    n = len(population)
    pareto_indices = []
    
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j:
                if (objectives[0, j] <= objectives[0, i] and 
                    objectives[1, j] <= objectives[1, i] and
                    (objectives[0, j] < objectives[0, i] or objectives[1, j] < objectives[1, i])):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    
    pareto_population = population[pareto_indices]
    print(f"  Carteras en frontera de Pareto: {len(pareto_population)} de {n}")
    
    return pareto_population, pareto_indices


def oos_analysis(population_train, returns_matrix_train, returns_matrix_val, algo_name, output_dir):
    """Analisis Out-of-Sample completo."""
    
    pareto_pop, _ = select_pareto_front(population_train, returns_matrix_train)
    
    # Calcular metricas en TRAINING
    train_mdd = []
    train_ret = []
    for w in pareto_pop:
        mdd, neg_ret = evaluate(w, returns_matrix_train)
        train_mdd.append(mdd * 100)
        train_ret.append(-neg_ret * 100)
    
    # Calcular metricas en VALIDATION (OOS)
    val_mdd = []
    val_ret = []
    val_duration = []
    val_final_return = []
    
    for w in pareto_pop:
        mdd, neg_ret = evaluate(w, returns_matrix_val)
        val_mdd.append(mdd * 100)
        val_ret.append(-neg_ret * 100)
        
        portfolio_returns = np.dot(w, returns_matrix_val)
        wealth = np.cumprod(1 + portfolio_returns)
        wealth = np.insert(wealth, 0, 1.0)
        running_max = np.maximum.accumulate(wealth)
        drawdowns = 1 - wealth / running_max
        
        is_dd = drawdowns > 0
        max_dur = 0
        curr_dur = 0
        for dd in is_dd:
            if dd:
                curr_dur += 1
                max_dur = max(max_dur, curr_dur)
            else:
                curr_dur = 0
        val_duration.append(max_dur)
        
        val_final_return.append((wealth[-1] - 1) * 100)
    
    # Seleccionar las 3 carteras representativas
    idx_min_mdd = np.argmin(train_mdd)
    idx_max_ret = np.argmax(train_ret)
    sortino_approx = [train_ret[i] / (train_mdd[i] + 1e-10) for i in range(len(train_ret))]
    idx_best_sortino = np.argmax(sortino_approx)
    
    # Crear tabla de resultados
    results = []
    for name, idx in [("Min MDD", idx_min_mdd), 
                       ("Best Sortino", idx_best_sortino),
                       ("Max Return", idx_max_ret)]:
        results.append({
            'Cartera': name,
            'MDD Train (%)': f"{train_mdd[idx]:.2f}",
            'MDD Val (%)': f"{val_mdd[idx]:.2f}",
            'Ret Train (%)': f"{train_ret[idx]:.2f}",
            'Ret Val (%)': f"{val_ret[idx]:.2f}",
            'Ret Acum Val (%)': f"{val_final_return[idx]:.2f}",
            'Dur DD Val (dias)': f"{val_duration[idx]:.0f}"
        })
    
    results_df = pd.DataFrame(results)
    
    # Guardar graficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].scatter(train_mdd, val_mdd, alpha=0.6, color='steelblue')
    axes[0].plot([min(train_mdd), max(train_mdd)], 
                 [min(train_mdd), max(train_mdd)], 
                 'r--', label='Mismo rendimiento')
    axes[0].set_xlabel('MDD Entrenamiento (%)')
    axes[0].set_ylabel('MDD Validacion (%)')
    axes[0].set_title(f'MDD: Train vs Validation - {algo_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(train_ret, val_ret, alpha=0.6, color='darkgreen')
    axes[1].plot([min(train_ret), max(train_ret)], 
                 [min(train_ret), max(train_ret)], 
                 'r--', label='Mismo rendimiento')
    axes[1].set_xlabel('Retorno Entrenamiento (%)')
    axes[1].set_ylabel('Retorno Validacion (%)')
    axes[1].set_title(f'Retorno: Train vs Validation - {algo_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(train_mdd, train_ret, alpha=0.7, color='blue', 
                    label='Train (Pareto)', s=30)
    axes[2].scatter(val_mdd, val_ret, alpha=0.7, color='red', 
                    label='Validation (OOS)', s=30, marker='x')
    axes[2].set_xlabel('MDD (%)')
    axes[2].set_ylabel('Retorno Medio (%)')
    axes[2].set_title(f'Frontera de Pareto OOS - {algo_name}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/oos_{algo_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Validacion Out-of-Sample')
    parser.add_argument('--prob', type=int, required=True, 
                        help='Numero de problema (1-4)')
    args = parser.parse_args()
    
    config = get_problem_config(args.prob)
    
    print("=" * 70)
    print(f"  VALIDACION OOS - {config['descripcion']}")
    print("=" * 70)
    
    data_dir = config['data_dir']
    results_parent_dir = f"results/{config['name']}"
    
    val_path = f'{data_dir}/val_returns.csv'
    train_path = f'{data_dir}/returns_matrix_train.npy'
    
    if not os.path.exists(val_path) or not os.path.exists(train_path):
        print(f"[ERROR] Datos no encontrados en {data_dir}")
        sys.exit(1)
    
    # Cargar datos
    print("\nCargando datos...")
    returns_matrix_val, val_returns_df = load_validation_data(val_path)
    returns_matrix_train = np.load(train_path)
    
    print(f"  Train: {returns_matrix_train.shape[0]} activos, {returns_matrix_train.shape[1]} dias")
    
    # Buscar poblaciones guardadas
    pattern = f"{results_parent_dir}/*_*/final_population.npy"
    pop_files = glob.glob(pattern)
    
    if len(pop_files) == 0:
        print(f"\n[ERROR] No se encontraron poblaciones en {results_parent_dir}")
        print(f"Ejecuta primero: python main.py --prob {args.prob}")
        sys.exit(1)
    
    print(f"\nSe encontraron {len(pop_files)} poblaciones:")
    for f in pop_files:
        print(f"  - {f}")
    
    # Crear directorio de salida
    output_dir = f"{results_parent_dir}/oos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar resultados en .txt
    txt_lines = []
    txt_lines.append("=" * 70)
    txt_lines.append(f"  VALIDACION OUT-OF-SAMPLE - {config['descripcion']}")
    txt_lines.append("=" * 70)
    txt_lines.append("")
    
    all_results = {}
    
    for pop_file in pop_files:
        # Extraer nombre del algoritmo de la ruta (funciona en Windows y Linux)
        folder_name = os.path.basename(os.path.dirname(pop_file))
        algo_name = folder_name.split('_')[0]
        print(f"\n{'='*60}")
        print(f"  Analizando: {algo_name}")
        print(f"{'='*60}")
        
        population = np.load(pop_file)
        results_df = oos_analysis(population, returns_matrix_train, 
                                  returns_matrix_val, algo_name, output_dir)
        
        print(f"\nResultados OOS para {algo_name}:")
        print(results_df.to_string(index=False))
        
        all_results[algo_name] = results_df
        
        # Guardar CSV
        results_df.to_csv(f'{output_dir}/oos_{algo_name}.csv', index=False)
        
        # Anadir al txt
        txt_lines.append(f"  Algoritmo: {algo_name}")
        txt_lines.append(f"  {'Cartera':<15} {'MDD Train':>10} {'MDD Val':>10} {'Ret Train':>10} {'Ret Val':>10} {'Ret Acum Val':>12} {'Dur DD':>8}")
        txt_lines.append("  " + "-" * 80)
        for _, row in results_df.iterrows():
            txt_lines.append(f"  {row['Cartera']:<15} {row['MDD Train (%)']:>10} {row['MDD Val (%)']:>10} {row['Ret Train (%)']:>10} {row['Ret Val (%)']:>10} {row['Ret Acum Val (%)']:>12} {row['Dur DD Val (dias)']:>8}")
        txt_lines.append("")
    
    # Resumen comparativo
    txt_lines.append("-" * 70)
    txt_lines.append("  RESUMEN COMPARATIVO OOS - CARTERA MIN MDD")
    txt_lines.append("-" * 70)
    txt_lines.append(f"  {'Algoritmo':<12} {'MDD Val':>10} {'Ret Acum Val':>12}")
    txt_lines.append("  " + "-" * 40)
    
    print(f"\n{'='*60}")
    print(f"  RESUMEN OOS - Cartera Min MDD")
    print(f"{'='*60}")
    
    for algo_name in sorted(all_results.keys()):
        row = all_results[algo_name][all_results[algo_name]['Cartera'] == 'Min MDD'].iloc[0]
        txt_lines.append(f"  {algo_name:<12} {row['MDD Val (%)']:>10} {row['Ret Acum Val (%)']:>12}")
        print(f"  {algo_name:<12} MDD Val: {row['MDD Val (%)']} | Ret Acum: {row['Ret Acum Val (%)']}")
    
    txt_lines.append("")
    txt_lines.append("=" * 70)
    
    # Guardar archivo .txt
    txt_path = f"{output_dir}/oos_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_lines))
    
    print(f"\n[OK] Resultados guardados en:")
    print(f"  - {txt_path}")
    print(f"  - {output_dir}/oos_*.png")
    print(f"  - {output_dir}/oos_*.csv")


if __name__ == "__main__":
    main()