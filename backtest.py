"""
Backtest Rolling para evaluar la estabilidad temporal de las carteras.
Compara los 5 algoritmos evolutivos: NSGA-II, SPEA2, NPGA2, PESA, e-MOEA.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import timedelta

from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.evaluation import precompute_objectives, evaluate


def run_rolling_for_algorithm(algo_name, algo_class, prices, train_days, test_days, step_days, algo_params):
    """
    Ejecuta backtest rolling para un algoritmo específico.
    
    Returns:
    - results_df: DataFrame con resultados de cada ventana
    """
    
    total_days = len(prices)
    results = []
    window = 0
    
    print(f"\n  Ejecutando rolling para {algo_name}...")
    
    for start in range(0, total_days - train_days - test_days + 1, step_days):
        window += 1
        train_end = start + train_days
        test_start = train_end
        test_end = test_start + test_days
        
        if test_end > total_days:
            break
        
        # Datos de entrenamiento
        train_prices = prices.iloc[start:train_end]
        train_returns = train_prices.pct_change().dropna()
        returns_matrix_train = train_returns.values.T
        
        # Datos de validación forward
        test_prices = prices.iloc[test_start:test_end]
        test_returns = test_prices.pct_change().dropna()
        returns_matrix_test = test_returns.values.T
        
        if returns_matrix_train.shape[1] < 10 or returns_matrix_test.shape[1] < 5:
            continue
        
        # Parámetros para esta ventana
        params = algo_params.copy()
        params['returns_matrix'] = returns_matrix_train
        params['num_assets'] = returns_matrix_train.shape[0]
        params['mutation_rate'] = 1 / params['num_assets']
        
        # Ejecutar algoritmo
        algo = algo_class(**params)
        population = algo.evolve()
        
        # Evaluar población en entrenamiento
        objectives = precompute_objectives(population, returns_matrix_train)
        
        # Cartera min MDD
        idx_min_mdd = np.argmin(objectives[0, :])
        w_min_mdd = population[idx_min_mdd]
        
        # Cartera max return
        idx_max_ret = np.argmin(objectives[1, :])
        w_max_ret = population[idx_max_ret]
        
        # Cartera best sortino aproximado
        sortinos = []
        for w in population:
            mdd, neg_ret = evaluate(w, returns_matrix_train)
            ret = -neg_ret
            sortinos.append(ret / (mdd + 1e-10))
        idx_best_sortino = np.argmax(sortinos)
        w_best_sortino = population[idx_best_sortino]
        
        # Evaluar en TEST
        for name, w in [('Min MDD', w_min_mdd), ('Best Sortino', w_best_sortino), ('Max Return', w_max_ret)]:
            mdd_test, neg_ret_test = evaluate(w, returns_matrix_test)
            ret_test = -neg_ret_test * 100
            mdd_test_pct = mdd_test * 100
            
            portfolio_returns = np.dot(w, returns_matrix_test)
            wealth = np.cumprod(1 + portfolio_returns)
            ret_acum = (wealth[-1] - 1) * 100
            
            results.append({
                'Algoritmo': algo_name,
                'Ventana': window,
                'Train_Start': prices.index[start].date(),
                'Train_End': prices.index[train_end-1].date(),
                'Test_Start': prices.index[test_start].date(),
                'Test_End': prices.index[test_end-1].date(),
                'Cartera': name,
                'MDD_Test_%': round(mdd_test_pct, 2),
                'Ret_Medio_Test_%': round(ret_test, 2),
                'Ret_Acum_Test_%': round(ret_acum, 2)
            })
        
        if window % 3 == 0:
            print(f"    Ventana {window} completada")
    
    results_df = pd.DataFrame(results)
    print(f"    Total ventanas: {window}")
    
    return results_df


def plot_rolling_comparison(all_results_df, output_dir='results/rolling'):
    """
    Gráficos comparativos de todos los algoritmos.
    """
    
    # Gráfico 1: Retorno acumulado total por algoritmo y cartera
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, cartera in zip(axes, ['Min MDD', 'Best Sortino', 'Max Return']):
        data = all_results_df[all_results_df['Cartera'] == cartera]
        algoritmos = data['Algoritmo'].unique()
        
        x = np.arange(len(algoritmos))
        width = 0.6
        
        ret_totales = []
        for algo in algoritmos:
            algo_data = data[data['Algoritmo'] == algo]
            ret_total = np.sum(algo_data['Ret_Acum_Test_%'].values)
            ret_totales.append(ret_total)
        
        colores = ['green' if r > 0 else 'red' for r in ret_totales]
        ax.bar(algoritmos, ret_totales, color=colores, alpha=0.7, width=width)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Retorno Acumulado Total (%)')
        ax.set_title(f'{cartera} - Retorno Total por Algoritmo')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
        
        for i, (algo, ret) in enumerate(zip(algoritmos, ret_totales)):
            ax.text(i, ret + (2 if ret > 0 else -5), f'{ret:.1f}%', 
                    ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_total_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Evolución acumulada de cada algoritmo (cartera Min MDD)
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    data_min_mdd = all_results_df[all_results_df['Cartera'] == 'Min MDD']
    algoritmos = data_min_mdd['Algoritmo'].unique()
    
    for algo in algoritmos:
        algo_data = data_min_mdd[data_min_mdd['Algoritmo'] == algo].sort_values('Ventana')
        cumulative = np.cumsum(algo_data['Ret_Acum_Test_%'].values)
        ax2.plot(algo_data['Ventana'], cumulative, marker='o', label=algo, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Ventana', fontsize=12)
    ax2.set_ylabel('Retorno Acumulado (%)', fontsize=12)
    ax2.set_title('Evolución del Retorno Acumulado - Cartera Min MDD (todos los algoritmos)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_cumulative_all_algorithms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3: % de ventanas positivas por algoritmo (Min MDD)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    pct_positivos = []
    for algo in algoritmos:
        algo_data = data_min_mdd[data_min_mdd['Algoritmo'] == algo]
        pct = np.mean(algo_data['Ret_Acum_Test_%'].values > 0) * 100
        pct_positivos.append(pct)
    
    ax3.bar(algoritmos, pct_positivos, color='steelblue', alpha=0.7)
    ax3.set_ylabel('% Ventanas Positivas', fontsize=12)
    ax3.set_title('Fiabilidad: % de Ventanas con Retorno Positivo - Cartera Min MDD', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15)
    
    for i, (algo, pct) in enumerate(zip(algoritmos, pct_positivos)):
        ax3.text(i, pct + 2, f'{pct:.0f}%', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_reliability.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_rolling_results(all_results_df, output_dir='results/rolling', total_ventanas=7):
    """
    Guarda resultados detallados en archivo .txt
    """
    
    data_min_mdd = all_results_df[all_results_df['Cartera'] == 'Min MDD']
    data_sortino = all_results_df[all_results_df['Cartera'] == 'Best Sortino']
    data_max_ret = all_results_df[all_results_df['Cartera'] == 'Max Return']
    algoritmos = sorted(data_min_mdd['Algoritmo'].unique())
    
    lines = []
    lines.append("=" * 70)
    lines.append("  RESULTADOS BACKTEST ROLLING - TODOS LOS ALGORITMOS")
    lines.append("=" * 70)
    lines.append(f"  Ventanas totales: {total_ventanas}")
    lines.append(f"  Algoritmos evaluados: {', '.join(algoritmos)}")
    lines.append("")
    
    # Tabla comparativa para Min MDD
    lines.append("-" * 70)
    lines.append("  CARTERA: MÍNIMO MDD")
    lines.append("-" * 70)
    lines.append(f"  {'Algoritmo':<12} {'Ret Medio':>10} {'% Vent +':>10} {'Ret Total':>10} {'MDD Medio':>10}")
    lines.append("  " + "-" * 55)
    
    for algo in algoritmos:
        algo_data = data_min_mdd[data_min_mdd['Algoritmo'] == algo]
        ret_medio = np.mean(algo_data['Ret_Acum_Test_%'].values)
        pct_pos = np.mean(algo_data['Ret_Acum_Test_%'].values > 0) * 100
        ret_total = np.sum(algo_data['Ret_Acum_Test_%'].values)
        mdd_medio = np.mean(algo_data['MDD_Test_%'].values)
        lines.append(f"  {algo:<12} {ret_medio:>8.2f}% {pct_pos:>8.0f}% {ret_total:>10.2f}% {mdd_medio:>8.2f}%")
    
    lines.append("")
    
    # Tabla comparativa para Best Sortino
    lines.append("-" * 70)
    lines.append("  CARTERA: BEST SORTINO")
    lines.append("-" * 70)
    lines.append(f"  {'Algoritmo':<12} {'Ret Medio':>10} {'% Vent +':>10} {'Ret Total':>10} {'MDD Medio':>10}")
    lines.append("  " + "-" * 55)
    
    for algo in algoritmos:
        algo_data = data_sortino[data_sortino['Algoritmo'] == algo]
        ret_medio = np.mean(algo_data['Ret_Acum_Test_%'].values)
        pct_pos = np.mean(algo_data['Ret_Acum_Test_%'].values > 0) * 100
        ret_total = np.sum(algo_data['Ret_Acum_Test_%'].values)
        mdd_medio = np.mean(algo_data['MDD_Test_%'].values)
        lines.append(f"  {algo:<12} {ret_medio:>8.2f}% {pct_pos:>8.0f}% {ret_total:>10.2f}% {mdd_medio:>8.2f}%")
    
    lines.append("")
    
    # Tabla comparativa para Max Return
    lines.append("-" * 70)
    lines.append("  CARTERA: MÁXIMO RETORNO")
    lines.append("-" * 70)
    lines.append(f"  {'Algoritmo':<12} {'Ret Medio':>10} {'% Vent +':>10} {'Ret Total':>10} {'MDD Medio':>10}")
    lines.append("  " + "-" * 55)
    
    for algo in algoritmos:
        algo_data = data_max_ret[data_max_ret['Algoritmo'] == algo]
        ret_medio = np.mean(algo_data['Ret_Acum_Test_%'].values)
        pct_pos = np.mean(algo_data['Ret_Acum_Test_%'].values > 0) * 100
        ret_total = np.sum(algo_data['Ret_Acum_Test_%'].values)
        mdd_medio = np.mean(algo_data['MDD_Test_%'].values)
        lines.append(f"  {algo:<12} {ret_medio:>8.2f}% {pct_pos:>8.0f}% {ret_total:>10.2f}% {mdd_medio:>8.2f}%")
    
    lines.append("")
    
    # Ranking final
    lines.append("-" * 70)
    lines.append("  RANKING FINAL - CARTERA MIN MDD")
    lines.append("-" * 70)
    
    ranking = []
    for algo in algoritmos:
        algo_data = data_min_mdd[data_min_mdd['Algoritmo'] == algo]
        ret_total = np.sum(algo_data['Ret_Acum_Test_%'].values)
        ranking.append((algo, ret_total))
    
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    for i, (algo, ret) in enumerate(ranking):
        medalla = ['1.', '2.', '3.', '4.', '5.'][i]
        lines.append(f"  {medalla} {algo:<12} Retorno Total: {ret:>8.2f}%")
    
    lines.append("")
    lines.append("=" * 70)
    
    # Guardar archivo
    output_str = "\n".join(lines)
    
    with open(f'{output_dir}/rolling_results.txt', 'w') as f:
        f.write(output_str)
    
    print(output_str)


def main():
    """
    Backtest rolling para los 5 algoritmos.
    """
    
    print("=" * 70)
    print("  BACKTEST ROLLING - TODOS LOS ALGORITMOS")
    print("=" * 70)
    
    # Cargar precios
    prices = pd.read_csv('data/ibex35/all_prices.csv', index_col=0, parse_dates=True)
    
    total_days = len(prices)
    train_months = 6
    train_days = train_months * 21
    test_days = 20
    step_days = 20
    
    total_ventanas = (total_days - train_days - test_days) // step_days + 1
    
    print(f"Datos totales: {total_days} días")
    print(f"Ventana entrenamiento: {train_days} días ({train_months} meses)")
    print(f"Ventana validación: {test_days} días")
    print(f"Paso: {step_days} días")
    print(f"Total ventanas estimadas: {total_ventanas}")
    print(f"Algoritmos: NSGA-II, SPEA2, NPGA2, PESA, e-MOEA")
    
    # Parámetros base (reducidos para rolling más rápido)
    base_params = {
        'N_arc': 100,
        'N_pop': 100,
        'generations': 100,
        'cardinality': 6,
        'crossover_rate': 0.9,
    }
    
    # Parámetros específicos
    npga2_params = {'tdom': 10, 'rsh': 0.7}
    pesa_params = {'grid_divisions': 10}
    emoea_params = {'e': 0.00458 * 6}
    
    # Definir algoritmos a ejecutar
    algoritmos = [
        ('NSGA-II', NSGA2, {}),
        ('SPEA2', SPEA2, {}),
        ('NPGA2', NPGA2, npga2_params),
        ('PESA', PESA, pesa_params),
        ('e-MOEA', E_MOEA, emoea_params),
    ]
    
    all_results = []
    
    for algo_name, algo_class, extra_params in algoritmos:
        params = base_params.copy()
        params.update(extra_params)
        
        results_df = run_rolling_for_algorithm(
            algo_name, algo_class, prices, 
            train_days, test_days, step_days, params
        )
        
        all_results.append(results_df)
        
        print(f"    {algo_name}: {len(results_df)//3} ventanas completadas")
    
    # Combinar todos los resultados
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Crear directorio
    os.makedirs('results/rolling', exist_ok=True)
    
    # Guardar CSV detallado
    all_results_df.to_csv('results/rolling/rolling_all_results.csv', index=False)
    
    # Generar gráficos comparativos
    print(f"\nGenerando gráficos comparativos...")
    plot_rolling_comparison(all_results_df)
    
    # Guardar resumen .txt
    save_rolling_results(all_results_df, total_ventanas=total_ventanas)
    
    print(f"\n✅ Todos los resultados guardados en: results/rolling/")
    print(f"   - rolling_results.txt")
    print(f"   - rolling_all_results.csv")
    print(f"   - rolling_total_comparison.png")
    print(f"   - rolling_cumulative_all_algorithms.png")
    print(f"   - rolling_reliability.png")


if __name__ == "__main__":
    main()