"""
Script para descargar datos históricos del IBEX 35 y prepararlos para el TFG.
Training: 1 año
Validación: 1 mes adicional
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ============================================
# 1. Definir fechas
# ============================================
end_date = datetime.now()
start_date = end_date - timedelta(days=395)  # ~13 meses

print(f"Descargando datos desde {start_date.date()} hasta {end_date.date()}")

# ============================================
# 2. Descargar acciones del IBEX 35
# ============================================
ibex_tickers = [
    'SAN.MC',   # Banco Santander
    'BBVA.MC',  # BBVA
    'TEF.MC',   # Telefónica
    'IBE.MC',   # Iberdrola
    'ITX.MC',   # Inditex
    'REP.MC',   # Repsol
]

print("Descargando acciones del IBEX 35...")
raw_data = yf.download(ibex_tickers, start=start_date, end=end_date)

# Manejar formato de yfinance (puede variar según versión)
if isinstance(raw_data.columns, pd.MultiIndex):
    # Formato nuevo: MultiIndex (Price, Ticker)
    if 'Adj Close' in raw_data.columns.levels[0]:
        stock_data = raw_data['Adj Close'].copy()
    else:
        stock_data = raw_data['Close'].copy()
else:
    # Formato antiguo: columnas planas
    stock_data = raw_data.copy()

# Eliminar columnas con todo NaN y llenar huecos
stock_data = stock_data.dropna(axis=1, how='all')
stock_data = stock_data.ffill()  # Forward fill para días sin cotización

print(f"Acciones descargadas: {list(stock_data.columns)}")
print(f"Período: {stock_data.index[0].date()} a {stock_data.index[-1].date()}")
print(f"Días totales: {len(stock_data)}")

# ============================================
# 3. Separar training y validación
# ============================================
cutoff_date = stock_data.index[-1] - timedelta(days=30)

train_data = stock_data[stock_data.index < cutoff_date].copy()
val_data = stock_data[stock_data.index >= cutoff_date].copy()

print(f"\nEntrenamiento: {train_data.index[0].date()} a {train_data.index[-1].date()} ({len(train_data)} días)")
print(f"Validación: {val_data.index[0].date()} a {val_data.index[-1].date()} ({len(val_data)} días)")

# ============================================
# 4. Rendimientos diarios
# ============================================
train_returns = train_data.pct_change().dropna()
val_returns = val_data.pct_change().dropna()

# Quitar columnas con NaN
train_returns = train_returns.dropna(axis=1)
val_returns = val_returns.dropna(axis=1)

# Acciones comunes
common_assets = train_returns.columns.intersection(val_returns.columns)
train_returns = train_returns[common_assets]
val_returns = val_returns[common_assets]

print(f"\nActivos finales: {len(common_assets)}")
for a in common_assets:
    print(f"  - {a}")

# ============================================
# 5. Matriz (n_assets, n_periods)
# ============================================
returns_matrix_train = train_returns.values.T

print(f"\nMatriz de entrenamiento: {returns_matrix_train.shape}")
print(f"  Activos: {returns_matrix_train.shape[0]}")
print(f"  Días: {returns_matrix_train.shape[1]}")

# ============================================
# 6. Guardar
# ============================================
output_dir = 'data/ibex35'
os.makedirs(output_dir, exist_ok=True)

np.save(f'{output_dir}/returns_matrix_train.npy', returns_matrix_train)
train_returns.to_csv(f'{output_dir}/train_returns.csv')
val_returns.to_csv(f'{output_dir}/val_returns.csv')
stock_data.to_csv(f'{output_dir}/all_prices.csv')

with open(f'{output_dir}/info.txt', 'w') as f:
    f.write(f"IBEX 35 Data for TFG\n")
    f.write(f"{'='*50}\n")
    f.write(f"Training: {train_returns.index[0].date()} to {train_returns.index[-1].date()}\n")
    f.write(f"Validation: {val_returns.index[0].date()} to {val_returns.index[-1].date()}\n")
    f.write(f"Assets: {len(common_assets)}\n")
    f.write(f"Training days: {len(train_returns)}\n")
    f.write(f"Validation days: {len(val_returns)}\n")
    for a in common_assets:
        f.write(f"  - {a}\n")

print(f"\n✅ Datos guardados en '{output_dir}/'")