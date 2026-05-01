Este repositorio contiene la implementación en Python de cinco algoritmos evolutivos multiobjetivo (MOEAs) para resolver el problema de optimización de porfolios con restricciones de cardinalidad (MVCCPO). Los algoritmos evaluados incluyen:

- NSGA-II  
- SPEA2  
- NPGA2  
- PESA  
- e-MOEA  

Se han realizado experimentos comparativos sobre datos financieros reales y sintéticos para evaluar su desempeño.

## Descripción de carpetas y archivos

- **`algorithms/`**: Implementaciones de los algoritmos evolutivos.
- **`algorithms/utils/`**: Funciones auxiliares para:
  - carga y preprocesamiento de datos
  - evaluación de individuos
  - operadores genéticos
  - visualización de resultados
- **`data/`**: Datos financieros reales y sintéticos.
- **`results/`**: Métricas de evaluación e imágenes de las fronteras de Pareto.
- **`main.py`**: Script principal de ejecución de experimentos.
- **`requirements.txt`**: Dependencias del proyecto.
- **`README.md`**: Documentación general del repositorio.

## Instrucciones de uso

### 1. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate   # En Linux/macOS
venv\Scripts\activate      # En Windows
```
### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el script principal de ejemplo
```bash
python main.py
```
Nota: Dentro de main.py, puedes elegir qué algoritmo ejecutar comentando o descomentando las líneas correspondientes, además de elegir los parámetros de todos los algoritmos.
Cada algoritmo, proporciona la población final, las métricas de evaluación (hipervolumen e índice de Sharpe medio) si se llama a la función encargada de ello, tiempo computacional, el número de individuos por cada cardinal de activo permitido y un plot con la población final, la frontera de Pareto.

#### Consideraciones
- **Aleatoriedad**: Debido a la naturaleza estocástica de los algoritmos evolutivos, los resultados pueden variar entre ejecuciones.
- **Modularidad**: El código está diseñado para facilitar la adición de nuevos algoritmos o configuraciones experimentales.
