"""
Script para generar 4 datasets diferentes del IBEX 35 para el TFG.
Cada dataset: 1 año de entrenamiento + 1 mes de validacion.

Problema 1: 10 empresas (Grupo A) - Entrenamiento 2025, Validacion Enero 2026
Problema 2: 10 empresas (Grupo A) - Entrenamiento 2024, Validacion Enero 2025
Problema 3: 10 empresas (Grupo B) - Entrenamiento 2025, Validacion Enero 2026
Problema 4: 10 empresas (Grupo B) - Entrenamiento 2024, Validacion Enero 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ============================================
# CONFIGURACION DE LOS 4 PROBLEMAS
# ============================================

# Grupo A: 10 empresas grandes/medianas del IBEX 35
GRUPO_A = [
    'SAN.MC',   # Banco Santander
    'BBVA.MC',  # BBVA
    'TEF.MC',   # Telefonica
    'IBE.MC',   # Iberdrola
    'ITX.MC',   # Inditex
    'REP.MC',   # Repsol
    'AMS.MC',   # Amadeus
    'FER.MC',   # Ferrovial
    'CABK.MC',  # Caixabank
    'CLNX.MC',  # Cellnex
]

# Grupo B: 10 empresas diferentes del IBEX 35
GRUPO_B = [
    'ACS.MC',   # ACS
    'AENA.MC',  # Aena
    'ANA.MC',   # Acciona
    'ENG.MC',   # Enagas
    'GRF.MC',   # Grifols
    'MAP.MC',   # Mapfre
    'MEL.MC',   # Melia Hotels
    'NTGY.MC',  # Naturgy
    'RED.MC',   # Redeia
    'SAB.MC',   # Banco Sabadell
]

# Definicion de los 4 problemas
PROBLEMAS = [
    {
        'id': 'prob1',
        'nombre': 'Grupo A - 2025',
        'tickers': GRUPO_A,
        'train_start': '2025-01-01',
        'train_end': '2025-12-31',
        'val_start': '2026-01-01',
        'val_end': '2026-01-31',
    },
    {
        'id': 'prob2',
        'nombre': 'Grupo A - 2024',
        'tickers': GRUPO_A,
        'train_start': '2024-01-01',
        'train_end': '2024-12-31',
        'val_start': '2025-01-01',
        'val_end': '2025-01-31',
    },
    {
        'id': 'prob3',
        'nombre': 'Grupo B - 2025',
        'tickers': GRUPO_B,
        'train_start': '2025-01-01',
        'train_end': '2025-12-31',
        'val_start': '2026-01-01',
        'val_end': '2026-01-31',
    },
    {
        'id': 'prob4',
        'nombre': 'Grupo B - 2024',
        'tickers': GRUPO_B,
        'train_start': '2024-01-01',
        'train_end': '2024-12-31',
        'val_start': '2025-01-01',
        'val_end': '2025-01-31',
    },
]


def download_data(tickers, train_start, train_end, val_start, val_end):
    """
    Descarga datos para un problema especifico.
    """
    # Descargar datos completos (train + validacion)
    print(f"  Descargando {len(tickers)} activos desde {train_start} hasta {val_end}...")
    
    raw_data = yf.download(tickers, start=train_start, end=val_end)
    
    # Manejar formato de yfinance
    if isinstance(raw_data.columns, pd.MultiIndex):
        if 'Adj Close' in raw_data.columns.levels[0]:
            stock_data = raw_data['Adj Close'].copy()
        else:
            stock_data = raw_data['Close'].copy()
    else:
        stock_data = raw_data.copy()
    
    # Limpiar datos
    stock_data = stock_data.dropna(axis=1, how='all')
    stock_data = stock_data.ffill()
    
    # Verificar que hay datos suficientes
    if stock_data.empty or len(stock_data) < 20:
        print(f"  [ERROR] No se pudieron descargar datos suficientes")
        return None, None, None
    
    # Separar train y validacion
    train_data = stock_data[stock_data.index <= train_end].copy()
    val_data = stock_data[stock_data.index >= val_start].copy()
    
    # Calcular rendimientos
    train_returns = train_data.pct_change().dropna()
    val_returns = val_data.pct_change().dropna()
    
    # Eliminar columnas con NaN
    train_returns = train_returns.dropna(axis=1)
    val_returns = val_returns.dropna(axis=1)
    
    # Mantener solo activos comunes
    common_assets = train_returns.columns.intersection(val_returns.columns)
    
    if len(common_assets) < 3:
        print(f"  [ERROR] Muy pocos activos con datos completos: {len(common_assets)}")
        return None, None, None
    
    train_returns = train_returns[common_assets]
    val_returns = val_returns[common_assets]
    
    print(f"  Activos validos: {len(common_assets)}")
    print(f"  Train: {train_returns.index[0].date()} a {train_returns.index[-1].date()} ({len(train_returns)} dias)")
    print(f"  Val: {val_returns.index[0].date()} a {val_returns.index[-1].date()} ({len(val_returns)} dias)")
    
    return train_returns, val_returns, common_assets


def save_problem(prob_id, train_returns, val_returns, common_assets, prob_nombre):
    """
    Guarda los datos de un problema en su carpeta correspondiente.
    """
    output_dir = f'data/{prob_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Matriz de entrenamiento (n_assets, n_periods)
    returns_matrix_train = train_returns.values.T
    
    # Guardar archivos
    np.save(f'{output_dir}/returns_matrix_train.npy', returns_matrix_train)
    train_returns.to_csv(f'{output_dir}/train_returns.csv')
    val_returns.to_csv(f'{output_dir}/val_returns.csv')
    
    # Guardar precios completos
    all_prices = pd.concat([train_returns, val_returns])
    all_prices.to_csv(f'{output_dir}/all_returns.csv')
    
    # Guardar info
    with open(f'{output_dir}/info.txt', 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {prob_nombre}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Training: {train_returns.index[0].date()} to {train_returns.index[-1].date()}\n")
        f.write(f"Validation: {val_returns.index[0].date()} to {val_returns.index[-1].date()}\n")
        f.write(f"Assets: {len(common_assets)}\n")
        f.write(f"Training days: {len(train_returns)}\n")
        f.write(f"Validation days: {len(val_returns)}\n")
        f.write(f"\nAssets:\n")
        for a in common_assets:
            f.write(f"  - {a}\n")
    
    print(f"  Datos guardados en '{output_dir}/'")


def main():
    """
    Genera los 4 datasets para el TFG.
    """
    print("=" * 70)
    print("  GENERANDO 4 DATASETS PARA EL TFG")
    print("=" * 70)
    
    resultados = []
    
    for prob in PROBLEMAS:
        print(f"\n{'='*70}")
        print(f"  {prob['id'].upper()}: {prob['nombre']}")
        print(f"{'='*70}")
        print(f"  Activos: {len(prob['tickers'])}")
        print(f"  Train: {prob['train_start']} a {prob['train_end']}")
        print(f"  Val: {prob['val_start']} a {prob['val_end']}")
        
        # Descargar datos
        train_returns, val_returns, common_assets = download_data(
            prob['tickers'],
            prob['train_start'],
            prob['train_end'],
            prob['val_start'],
            prob['val_end']
        )
        
        if train_returns is None:
            print(f"  [ERROR] Saltando {prob['id']} por falta de datos")
            continue
        
        # Guardar datos
        save_problem(prob['id'], train_returns, val_returns, common_assets, prob['nombre'])
        
        resultados.append({
            'id': prob['id'],
            'nombre': prob['nombre'],
            'activos': len(common_assets),
            'train_dias': len(train_returns),
            'val_dias': len(val_returns),
        })
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"  RESUMEN DE DATASETS GENERADOS")
    print(f"{'='*70}")
    print(f"  {'ID':<8} {'Nombre':<20} {'Activos':>8} {'Train Dias':>10} {'Val Dias':>10}")
    print(f"  {'-'*60}")
    
    for r in resultados:
        print(f"  {r['id']:<8} {r['nombre']:<20} {r['activos']:>8} {r['train_dias']:>10} {r['val_dias']:>10}")
    
    print(f"\n[OK] Datasets generados:")
    for r in resultados:
        print(f"  data/{r['id']}/ -> {r['nombre']} ({r['activos']} activos)")
    
    print(f"\n  Para usar un dataset en main.py, cambia la linea de carga a:")
    print(f"  returns_matrix = np.load('data/prob1/returns_matrix_train.npy')")


if __name__ == "__main__":
    main()