"""
Backtest Rolling para evaluar la estabilidad temporal de las carteras.
Uso: python rolling.py --prob 1
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.evaluation import precompute_objectives, evaluate


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


def run_rolling_for_algorithm(algo_name, algo_class, prices, train_days, test_days, step_days, algo_params):
    """Ejecuta backtest rolling para un algoritmo especifico."""
    
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

        train_returns = prices.iloc[start:train_end].dropna()
        returns_matrix_train = train_returns.values.T

        test_returns = prices.iloc[test_start:test_end].dropna()
        returns_matrix_test = test_returns.values.T
        
        if returns_matrix_train.shape[1] < 10 or returns_matrix_test.shape[1] < 5:
            continue
        
        # Limpiar datos de entrenamiento
        returns_matrix_train = np.nan_to_num(returns_matrix_train, nan=0.0, posinf=0.0, neginf=0.0)
        returns_matrix_train = np.clip(returns_matrix_train, -0.5, 0.5)
        
        # Limpiar datos de test
        returns_matrix_test = np.nan_to_num(returns_matrix_test, nan=0.0, posinf=0.0, neginf=0.0)
        returns_matrix_test = np.clip(returns_matrix_test, -0.5, 0.5)
        params = algo_params.copy()
        params['returns_matrix'] = returns_matrix_train
        params['num_assets'] = returns_matrix_train.shape[0]
        params['mutation_rate'] = 1 / params['num_assets']
        params['cardinality'] = params['num_assets']
        
        try:
            algo = algo_class(**params)
            population = algo.evolve()
        except Exception as e:
            print(f"    [ERROR] Ventana {window}: {str(e)[:50]}")
            continue
        
        objectives = precompute_objectives(population, returns_matrix_train)
        
        idx_min_mdd = np.argmin(objectives[0, :])
        w_min_mdd = population[idx_min_mdd]
        
        idx_max_ret = np.argmin(objectives[1, :])
        w_max_ret = population[idx_max_ret]
        
        sortinos = []
        for w in population:
            mdd, neg_ret = evaluate(w, returns_matrix_train)
            ret = -neg_ret
            sortinos.append(ret / (mdd + 1e-10))
        idx_best_sortino = np.argmax(sortinos)
        w_best_sortino = population[idx_best_sortino]
        
        for name, w in [('Min MDD', w_min_mdd), ('Best Sortino', w_best_sortino), ('Max Return', w_max_ret)]:
            mdd_test, neg_ret_test = evaluate(w, returns_matrix_test)
            ret_test = -neg_ret_test * 100
            mdd_test_pct = mdd_test * 100
            
            # Limpiar pesos
            w_clean = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w_clean = np.clip(w_clean, 0, 1)
            
            portfolio_returns = np.dot(w_clean, returns_matrix_test)
            portfolio_returns = np.nan_to_num(portfolio_returns, nan=0.0)
            portfolio_returns = np.clip(portfolio_returns, -0.5, 0.5)
            
            wealth = np.cumprod(1 + portfolio_returns)
            wealth = np.clip(wealth, 0.001, 100.0)
            ret_acum = (wealth[-1] - 1) * 100
            ret_acum = np.clip(ret_acum, -100, 1000)  # Entre -100% y +1000%
            if np.isnan(ret_acum) or np.isinf(ret_acum):
                ret_acum = 0.0
            
            results.append({
                'Algoritmo': algo_name,
                'Ventana': window,
                'Train_Start': prices.index[start].date(),
                'Train_End': prices.index[train_end-1].date(),
                'Test_Start': prices.index[test_start].date(),
                'Test_End': prices.index[test_end-1].date(),
                'Cartera': name,
                'MDD_Test_%': round(mdd_test_pct, 2) if not np.isnan(mdd_test_pct) else 0.0,
                'Ret_Medio_Test_%': round(ret_test, 2) if not np.isnan(ret_test) else 0.0,
                'Ret_Acum_Test_%': round(ret_acum, 2) if not np.isnan(ret_acum) else 0.0
            })
        
        if window % 3 == 0:
            print(f"    Ventana {window} completada")
    
    results_df = pd.DataFrame(results)
    print(f"    Total ventanas: {window}")
    
    return results_df


def plot_rolling_comparison(all_results_df, output_dir):
    """Graficos comparativos de todos los algoritmos."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Limpiar NaN e infinitos
    all_results_df = all_results_df.replace([np.inf, -np.inf], np.nan)
    all_results_df = all_results_df.fillna(0)
    
    # Recortar valores extremos (mas de 1000% es un error)
    all_results_df['Ret_Acum_Test_%'] = all_results_df['Ret_Acum_Test_%'].clip(-100, 100)
    all_results_df['MDD_Test_%'] = all_results_df['MDD_Test_%'].clip(0, 100)
    
    # Grafico 1: Retorno acumulado total
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, cartera in zip(axes, ['Min MDD', 'Best Sortino', 'Max Return']):
        data = all_results_df[all_results_df['Cartera'] == cartera]
        if len(data) == 0:
            ax.set_title(f'{cartera} - Sin datos')
            ax.set_ylim(-10, 10)
            continue
            
        algoritmos = sorted(data['Algoritmo'].unique())
        
        ret_totales = []
        for algo in algoritmos:
            algo_data = data[data['Algoritmo'] == algo]
            valores = algo_data['Ret_Acum_Test_%'].values
            valores = np.nan_to_num(valores, nan=0.0, posinf=0.0, neginf=0.0)
            valores = np.clip(valores, -50, 50)
            ret_total = np.sum(valores)
            ret_totales.append(ret_total)
        
        colores = ['green' if r > 0 else 'red' for r in ret_totales]
        ax.bar(algoritmos, ret_totales, color=colores, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Retorno Acumulado Total (%)')
        ax.set_title(f'{cartera} - Retorno Total por Algoritmo')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
        
        # Ajustar limites Y automaticamente
        max_abs = max(abs(r) for r in ret_totales) if ret_totales else 10
        ax.set_ylim(-max_abs * 1.5, max_abs * 1.5)
        
        for i, (algo, ret) in enumerate(zip(algoritmos, ret_totales)):
            offset = max_abs * 0.1
            ax.text(i, ret + offset, f'{ret:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_total_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Grafico 2: Evolucion acumulada Min MDD
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    data_min_mdd = all_results_df[all_results_df['Cartera'] == 'Min MDD']
    if len(data_min_mdd) > 0:
        algoritmos = sorted(data_min_mdd['Algoritmo'].unique())
        
        for algo in algoritmos:
            algo_data = data_min_mdd[data_min_mdd['Algoritmo'] == algo].sort_values('Ventana')
            valores = algo_data['Ret_Acum_Test_%'].values
            valores = np.nan_to_num(valores, nan=0.0, posinf=0.0, neginf=0.0)
            valores = np.clip(valores, -50, 50)
            cumulative = np.cumsum(valores)
            ax2.plot(algo_data['Ventana'].values, cumulative, marker='o', label=algo, linewidth=2, markersize=6)
        
        # Poner limites razonables
        y_min = min(0, np.min([np.min(np.cumsum(np.clip(np.nan_to_num(
            data_min_mdd[data_min_mdd['Algoritmo'] == a].sort_values('Ventana')['Ret_Acum_Test_%'].values,
            nan=0.0), -50, 50))) for a in algoritmos])) * 1.2
        y_max = max(0, np.max([np.max(np.cumsum(np.clip(np.nan_to_num(
            data_min_mdd[data_min_mdd['Algoritmo'] == a].sort_values('Ventana')['Ret_Acum_Test_%'].values,
            nan=0.0), -50, 50))) for a in algoritmos])) * 1.2
        ax2.set_ylim(y_min, y_max)
    
    ax2.set_xlabel('Ventana', fontsize=12)
    ax2.set_ylabel('Retorno Acumulado (%)', fontsize=12)
    ax2.set_title('Evolucion del Retorno Acumulado - Cartera Min MDD', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Grafico 3: Fiabilidad
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    if len(data_min_mdd) > 0:
        pct_positivos = []
        for algo in algoritmos:
            algo_data = data_min_mdd[data_min_mdd['Algoritmo'] == algo]
            valores = algo_data['Ret_Acum_Test_%'].values
            valores = np.nan_to_num(valores, nan=0.0)
            pct = np.mean(valores > 0) * 100
            pct_positivos.append(pct)
        
        ax3.bar(algoritmos, pct_positivos, color='steelblue', alpha=0.7)
        
        for i, (algo, pct) in enumerate(zip(algoritmos, pct_positivos)):
            ax3.text(i, pct + 2, f'{pct:.0f}%', ha='center', fontweight='bold', fontsize=11)
    
    ax3.set_ylabel('% Ventanas Positivas', fontsize=12)
    ax3.set_title('Fiabilidad - Cartera Min MDD', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 105)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_reliability.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_rolling_results(all_results_df, output_dir, config, total_ventanas):
    """Guarda resultados detallados en archivo .txt"""
    
    all_results_df = all_results_df.fillna(0)
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"  RESULTADOS BACKTEST ROLLING")
    lines.append(f"  {config['descripcion']}")
    lines.append("=" * 70)
    lines.append(f"  Ventanas totales: {total_ventanas}")
    lines.append("")
    
    for cartera in ['Min MDD', 'Best Sortino', 'Max Return']:
        data = all_results_df[all_results_df['Cartera'] == cartera]
        
        if len(data) == 0:
            continue
            
        algoritmos = sorted(data['Algoritmo'].unique())
        
        lines.append("-" * 70)
        lines.append(f"  CARTERA: {cartera.upper()}")
        lines.append("-" * 70)
        lines.append(f"  {'Algoritmo':<12} {'Ret Medio':>10} {'% Vent +':>10} {'Ret Total':>10}")
        lines.append("  " + "-" * 45)
        
        ranking = []
        for algo in algoritmos:
            algo_data = data[data['Algoritmo'] == algo]
            valores = np.nan_to_num(algo_data['Ret_Acum_Test_%'].values, nan=0.0)
            ret_medio = np.mean(valores)
            pct_pos = np.mean(valores > 0) * 100
            ret_total = np.sum(valores)
            
            lines.append(f"  {algo:<12} {ret_medio:>8.2f}% {pct_pos:>8.0f}% {ret_total:>10.2f}%")
            ranking.append((algo, ret_total))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"  Ranking: {', '.join([f'{i+1}.{a}({r:+.1f}%)' for i,(a,r) in enumerate(ranking)])}")
        lines.append("")
    
    lines.append("=" * 70)
    
    output_str = "\n".join(lines)
    
    with open(f'{output_dir}/rolling_results.txt', 'w', encoding='utf-8') as f:
        f.write(output_str)
    
    print(output_str)


def main():
    parser = argparse.ArgumentParser(description='Backtest Rolling')
    parser.add_argument('--prob', type=int, required=True, 
                        help='Numero de problema (1-4)')
    args = parser.parse_args()
    
    config = get_problem_config(args.prob)
    
    print("=" * 70)
    print(f"  BACKTEST ROLLING - {config['descripcion']}")
    print("=" * 70)
    
    data_dir = config['data_dir']
    prices_path = f'{data_dir}/all_returns.csv'
    
    if not os.path.exists(prices_path):
        print(f"[ERROR] No se encuentra {prices_path}")
        sys.exit(1)
    
    # Cargar precios
    print("\nCargando datos...")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    total_days = len(prices)
    train_months = 6
    train_days = train_months * 21
    test_days = 20
    step_days = 20
    
    total_ventanas = (total_days - train_days - test_days) // step_days + 1
    
    print(f"  Datos totales: {total_days} dias")
    print(f"  Ventana entrenamiento: {train_days} dias (~{train_months} meses)")
    print(f"  Ventana validacion: {test_days} dias")
    print(f"  Paso: {step_days} dias")
    print(f"  Total ventanas estimadas: {total_ventanas}")
    
    # Parametros base reducidos para rolling
    base_params = {
        'N_arc': 50,
        'N_pop': 50,
        'generations': 50,
        'cardinality': 10,
        'crossover_rate': 0.9,
    }
    
    npga2_params = {'tdom': 10, 'rsh': 0.7}
    pesa_params = {'grid_divisions': 10}
    emoea_params = {'e': 0.00458 * 6}
    
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
        
        if len(results_df) > 0:
            print(f"    {algo_name}: {len(results_df)//3} ventanas completadas")
    
    # Combinar resultados
    if len(all_results) > 0:
        all_results_df = pd.concat(all_results, ignore_index=True)
    else:
        print("\n[ERROR] No se generaron resultados")
        sys.exit(1)
    
    # Crear directorio de salida
    output_dir = f"results/{config['name']}/rolling"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar CSV
    all_results_df.to_csv(f'{output_dir}/rolling_all_results.csv', index=False)
    
    # Graficos
    print(f"\nGenerando graficos...")
    plot_rolling_comparison(all_results_df, output_dir)
    
    # Guardar resumen
    save_rolling_results(all_results_df, output_dir, config, total_ventanas)
    
    print(f"\n[OK] Resultados guardados en: {output_dir}/")
    print(f"   - rolling_results.txt")
    print(f"   - rolling_all_results.csv")
    print(f"   - rolling_total_comparison.png")
    print(f"   - rolling_cumulative.png")
    print(f"   - rolling_reliability.png")


if __name__ == "__main__":
    main()