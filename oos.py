"""
Validación Out-of-Sample para las carteras optimizadas con MDD.
Compara el rendimiento de las carteras en datos no vistos (1 mes adicional).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.utils.evaluation import evaluate

def load_validation_data(csv_path='data/ibex35/val_returns.csv'):
    """
    Carga los datos de validación (1 mes out-of-sample).
    """
    val_returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns_matrix_val = val_returns.values.T  # (n_assets, n_periods)
    
    print(f"Datos de validación: {returns_matrix_val.shape[0]} activos, "
          f"{returns_matrix_val.shape[1]} días")
    print(f"Período: {val_returns.index[0].date()} a {val_returns.index[-1].date()}")
    
    return returns_matrix_val, val_returns


def select_pareto_front(population, returns_matrix_train):
    """
    Selecciona solo las carteras no dominadas (frontera de Pareto) del entrenamiento.
    
    Returns:
    - pareto_population: Carteras no dominadas
    - pareto_indices: Índices de las carteras no dominadas
    """
    from algorithms.utils.evaluation import precompute_objectives
    
    objectives = precompute_objectives(population, returns_matrix_train)
    # objectives[0] = MDD, objectives[1] = -mean_return
    
    n = len(population)
    pareto_indices = []
    
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j:
                # j domina a i si es mejor o igual en ambos y estrictamente mejor en al menos uno
                if (objectives[0, j] <= objectives[0, i] and 
                    objectives[1, j] <= objectives[1, i] and
                    (objectives[0, j] < objectives[0, i] or objectives[1, j] < objectives[1, i])):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    
    pareto_population = population[pareto_indices]
    print(f"Carteras en la frontera de Pareto: {len(pareto_population)} de {n}")
    
    return pareto_population, pareto_indices


def oos_analysis(population_train, returns_matrix_train, returns_matrix_val, algo_name):
    """
    Análisis Out-of-Sample completo.
    
    Parameters:
    - population_train: Población final del entrenamiento
    - returns_matrix_train: Datos de entrenamiento
    - returns_matrix_val: Datos de validación (nuevos)
    - algo_name: Nombre del algoritmo
    
    Returns:
    - results_df: DataFrame con resultados OOS
    """
    
    # 1. Seleccionar frontera de Pareto del entrenamiento
    pareto_pop, pareto_idx = select_pareto_front(population_train, returns_matrix_train)
    
    # 2. Calcular métricas en TRAINING
    train_mdd = []
    train_ret = []
    for w in pareto_pop:
        mdd, neg_ret = evaluate(w, returns_matrix_train)
        train_mdd.append(mdd * 100)     # % MDD
        train_ret.append(-neg_ret * 100) # % retorno medio
    
    # 3. Calcular métricas en VALIDATION (OOS)
    val_mdd = []
    val_ret = []
    val_duration = []
    val_final_return = []  # Retorno acumulado total
    
    for w in pareto_pop:
        mdd, neg_ret = evaluate(w, returns_matrix_val)
        val_mdd.append(mdd * 100)
        val_ret.append(-neg_ret * 100)
        
        # Duración del drawdown en validación
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
        
        # Retorno acumulado final
        val_final_return.append((wealth[-1] - 1) * 100)  # % retorno total
    
    # 4. Seleccionar las 3 carteras más representativas
    # Cartera con menor MDD en entrenamiento
    idx_min_mdd = np.argmin(train_mdd)
    # Cartera con mayor retorno en entrenamiento
    idx_max_ret = np.argmax(train_ret)
    # Cartera intermedia (mejor Sortino aproximado en entrenamiento)
    sortino_approx = [train_ret[i] / (train_mdd[i] + 1e-10) for i in range(len(train_ret))]
    idx_best_sortino = np.argmax(sortino_approx)
    
    # 5. Crear tabla de resultados
    results = []
    for name, idx in [("Min MDD", idx_min_mdd), 
                       ("Max Return", idx_max_ret), 
                       ("Best Sortino", idx_best_sortino)]:
        results.append({
            'Cartera': name,
            'MDD Train (%)': f"{train_mdd[idx]:.2f}",
            'MDD Val (%)': f"{val_mdd[idx]:.2f}",
            'Ret Train (%)': f"{train_ret[idx]:.2f}",
            'Ret Val (%)': f"{val_ret[idx]:.2f}",
            'Ret Acum Val (%)': f"{val_final_return[idx]:.2f}",
            'Dur DD Val (días)': f"{val_duration[idx]:.0f}"
        })
    
    results_df = pd.DataFrame(results)
    
    # 6. Visualizar resultados OOS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gráfico 1: MDD Train vs Val
    axes[0].scatter(train_mdd, val_mdd, alpha=0.6, color='steelblue')
    axes[0].plot([min(train_mdd), max(train_mdd)], 
                 [min(train_mdd), max(train_mdd)], 
                 'r--', label='Mismo rendimiento')
    axes[0].set_xlabel('MDD Entrenamiento (%)')
    axes[0].set_ylabel('MDD Validación (%)')
    axes[0].set_title(f'MDD: Train vs Validation - {algo_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Retorno Train vs Val
    axes[1].scatter(train_ret, val_ret, alpha=0.6, color='darkgreen')
    axes[1].plot([min(train_ret), max(train_ret)], 
                 [min(train_ret), max(train_ret)], 
                 'r--', label='Mismo rendimiento')
    axes[1].set_xlabel('Retorno Entrenamiento (%)')
    axes[1].set_ylabel('Retorno Validación (%)')
    axes[1].set_title(f'Retorno: Train vs Validation - {algo_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gráfico 3: Frontera de Pareto Train vs puntos OOS
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
    plt.savefig(f'results/oos_validation_{algo_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results_df


def main():
    """
    Ejecutar validación OOS para todos los algoritmos evaluados.
    """
    import os
    os.makedirs('results', exist_ok=True)
    
    # Cargar datos de validación
    print("="*60)
    print("  VALIDACIÓN OUT-OF-SAMPLE")
    print("="*60)
    
    returns_matrix_val, val_returns_df = load_validation_data('data/ibex35/val_returns.csv')
    returns_matrix_train = np.load('data/ibex35/returns_matrix_train.npy')
    
    # Buscar archivos de población final en results/
    import glob
    pop_files = glob.glob('results/*_*/final_population_*.npy')
    
    if len(pop_files) == 0:
        print("\n⚠️ No se encontraron archivos de población final.")
        print("Primero ejecuta main.py con los algoritmos para generar las poblaciones.")
        print("Buscando en results/experiment_*/final_population_*.npy")
        return
    
    print(f"\nSe encontraron {len(pop_files)} archivos de población:")
    for f in pop_files:
        print(f"  - {f}")
    
    all_results = {}
    
    for pop_file in pop_files:
        algo_name = pop_file.split('final_population_')[1].replace('.npy', '')
        print(f"\n{'='*60}")
        print(f"  Analizando: {algo_name}")
        print(f"{'='*60}")
        
        population = np.load(pop_file)
        results_df = oos_analysis(population, returns_matrix_train, 
                                  returns_matrix_val, algo_name)
        
        print(f"\nResultados OOS para {algo_name}:")
        print(results_df.to_string(index=False))
        
        all_results[algo_name] = results_df
        
        # Guardar CSV
        results_df.to_csv(f'results/oos_{algo_name}.csv', index=False)
    
    # Tabla resumen comparativa
    print(f"\n{'='*60}")
    print(f"  RESUMEN COMPARATIVO OOS (Cartera Best Sortino)")
    print(f"{'='*60}")
    
    print(f"{'Algoritmo':<12} {'MDD Train':>10} {'MDD Val':>10} {'Ret Train':>10} {'Ret Val':>10} {'Ret Acum Val':>12}")
    print("-"*60)
    
    for algo_name, df in all_results.items():
        row = df[df['Cartera'] == 'Best Sortino'].iloc[0]
        print(f"{algo_name:<12} {row['MDD Train (%)']:>10} {row['MDD Val (%)']:>10} "
              f"{row['Ret Train (%)']:>10} {row['Ret Val (%)']:>10} {row['Ret Acum Val (%)']:>12}")


if __name__ == "__main__":
    main()