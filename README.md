# TFG - Minimización del Máximo Drawdown en Carteras de Inversión mediante Optimización Multiobjetivo

Este repositorio contiene la implementación en Python de cinco algoritmos evolutivos multiobjetivo (MOEAs) adaptados para resolver el problema de optimización de carteras mediante la **minimización del máximo drawdown (MDD)** y la **maximización del retorno medio esperado**.

Los algoritmos evaluados son:

- NSGA-II
- SPEA2
- NPGA2
- PESA
- e-MOEA

Los experimentos se han realizado sobre datos reales del **IBEX 35**, utilizando cuatro configuraciones distintas:

- Dos grupos de activos
- Dos períodos temporales

---

## Estructura del proyecto

### `algorithms/`
Implementaciones de los cinco algoritmos evolutivos.

### `algorithms/utils/`
Funciones auxiliares para:

- Carga y preprocesamiento de datos (`data_loader.py`)
- Evaluación de individuos con MDD (`evaluation.py`)
- Operadores genéticos y proyección al simplex:
  - `operators.py`
  - `projection.py`
  - `initialization.py`
- Métricas de rendimiento:
  - `performance.py`
  - `fitness.py`
- Visualización de resultados (`visualization.py`)
- Normalización de objetivos (`normalization.py`)

### `algorithms/tests/`
Scripts de validación:

- `oos.py`: validación *Out-of-Sample*
- `rolling.py`: *backtesting rolling* con ventanas deslizantes
- `networth_validation.py`: simulación de evolución de capital desde 1000 EUR

### `data/`
Datos financieros del IBEX 35 organizados en cuatro problemas (`prob1` a `prob4`), cada uno con conjuntos de entrenamiento y validación.

### `results/`
Resultados organizados por problema y algoritmo:

- Métricas
- Fronteras de Pareto
- Validaciones

### Otros archivos

- `prepare_ibex_data.py`: descarga y preparación de datasets desde Yahoo Finance
- `main.py`: script principal de ejecución
- `requirements.txt`: dependencias del proyecto
- `README.md`: documentación del repositorio

---

## Instrucciones de uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/carlosgmOY/TFG_Drawdown.git
cd TFG_Drawdown
```

### 2. Crear un entorno virtual (opcional pero recomendado)

**Linux/macOS**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Preparar los datos

```bash
python prepare_ibex_data.py
```

Este script:

- Descarga los datos históricos del IBEX 35 desde Yahoo Finance
- Genera cuatro configuraciones experimentales:
  - `data/prob1/`
  - `data/prob2/`
  - `data/prob3/`
  - `data/prob4/`

### 5. Ejecutar un experimento

El script principal acepta un parámetro `--prob` para seleccionar el problema (`1–4`).

Dentro de `main.py`, descomenta el algoritmo que quieras ejecutar.

Ejemplo:

```bash
python main.py --prob 1
```

### 6. Ejecutar validaciones

Una vez generadas las poblaciones mediante `main.py`, pueden ejecutarse las validaciones:

```bash
python algorithms/tests/oos.py --prob 1
python algorithms/tests/rolling.py --prob 1
python algorithms/tests/networth_validation.py --prob 1
```

---

## Métricas de evaluación

Cada algoritmo proporciona las siguientes métricas:

- **Hipervolumen**
  - Calidad de la frontera de Pareto

- **Sortino Ratio**
  - Rentabilidad ajustada al riesgo bajista

- **MDD medio y duración del drawdown**
  - Riesgo y persistencia de pérdidas

- **Tiempo de ejecución**

- **Frontera de Pareto**
  - Gráfico MDD vs Retorno

- **Distribución de activos por cartera**

---

## Consideraciones

### Aleatoriedad

Debido a la naturaleza estocástica de los algoritmos evolutivos, los resultados pueden variar entre ejecuciones.

### Modularidad

El código está diseñado para facilitar la incorporación de:

- Nuevos algoritmos
- Nuevas métricas
- Configuraciones experimentales adicionales

### Datos

Los datasets se generan a partir de Yahoo Finance.

Si algún *ticker* deja de estar disponible, modifica las listas:

- `GRUPO_A`
- `GRUPO_B`

dentro del archivo:

```python
prepare_ibex_data.py
```

---

## Autor

Trabajo Fin de Grado (TFG) centrado en optimización multiobjetivo aplicada a carteras de inversión con restricciones de riesgo basadas en *Maximum Drawdown*.