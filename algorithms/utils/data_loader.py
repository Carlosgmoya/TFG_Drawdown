import pandas as pd
import numpy as np

# This module contains functions to load datasets for portfolio optimization.

def load_dataset(file_path):
    """Load dataset and return expected returns and covariance matrix using only pandas."""

    # Read the entire file as lines using pandas
    raw_lines = pd.read_csv(file_path, header=None, dtype=str).squeeze().tolist()
    
    # Get number of assets
    num_assets = int(raw_lines[0])

    # Parse expected returns and std deviations
    stats_df = pd.read_csv(
        file_path, 
        sep=r'\s+', 
        skiprows=1, 
        nrows=num_assets, 
        header=None, 
        names=['return', 'std_dev']
    ).astype(float)

    # Initialize correlation matrix
    corr_matrix = pd.DataFrame(1.0, index=range(num_assets), columns=range(num_assets))

    # Read correlation data from the file
    correlations = pd.read_csv(
        file_path,
        sep=r'\s+',
        skiprows=num_assets + 1,
        header=None,
        names=['i', 'j', 'corr']
    )

    # Fill correlation matrix
    for _, row in correlations.iterrows():
        i_idx, j_idx = int(row['i']) - 1, int(row['j']) - 1
        corr = float(row['corr'])
        corr_matrix.iat[i_idx, j_idx] = corr
        corr_matrix.iat[j_idx, i_idx] = corr

    # Compute covariance matrix
    std_devs = stats_df['std_dev']
    std_outer = std_devs.to_frame().dot(std_devs.to_frame().T)
    cov_matrix = std_outer * corr_matrix

    return stats_df['return'].values, cov_matrix.values


def load_real_data(file_path):
    """Load real data from a CSV file and return expected returns and covariance matrix."""
    
    df = pd.read_csv(file_path, header=None, skiprows=1, dtype=str)

    # Remove the first column (dates)
    df = df.drop(columns=[0])

    # Convert all values to float, errors as strings are converted to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Remove columns that contain -99.99 or -999
    columnas_validas = df.columns[~df.isin([-99.99, -999]).any()]
    df_limpio = df[columnas_validas]

    print(df_limpio)

    # Calculate the average return vector, ignoring NaNs
    retuns = df_limpio.mean()

    # Convert vector to np.array
    retuns = retuns.to_numpy()

    # Calculate covariance matrix
    cov_matrix = df_limpio.cov() / 100
    # Convert matrix to np.array
    cov_matrix = cov_matrix.to_numpy()

    print(f"Expected Returns: {retuns}")
    sum(retuns)  # This line is not necessary, but it can be used to check the sum of returns
    print(f"Expected Returns Sum: {sum(retuns)}")
    print(f"Covariance Matrix:\n{cov_matrix}")

    return retuns, cov_matrix


def save_data_txt(file_path, output_path):
    """Load real data from a CSV file and save it in a specific text format."""
    # Load the real data
    returns, cov_matrix = load_real_data(file_path)

    # Number of assets
    N = len(returns)

    # Standard deviations
    stds = np.sqrt(np.diag(cov_matrix))

    # Correlation matrix
    corr_matrix = cov_matrix / np.outer(stds, stds)

    # Write to output file
    with open(output_path, 'w') as f:
        f.write(f"{N}\n")
        
        for i in range(N):
            f.write(f"{returns[i]:.6f} {stds[i]:.6f}\n")

        for i in range(N):
            for j in range(i, N):
                f.write(f"{i+1} {j+1} {corr_matrix[i, j]:.6f}\n")

    print(f"File saved in: {output_path}")


def load_dataset_drawdown(file_path):

    df = pd.read_csv(file_path)

    # Quitar columna de fechas si existe
    if not np.issubdtype(df.iloc[:,0].dtype, np.number):
        df = df.drop(columns=[df.columns[0]])

    # Convertir a numérico
    df = df.apply(pd.to_numeric, errors='coerce')

    # Eliminar valores problemáticos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ❗ IMPORTANTE: eliminar filas con NaN
    df = df.dropna()

    # ❗ IMPORTANTE: eliminar ceros (precios inválidos)
    df = df[(df > 0).all(axis=1)]

    data = df.to_numpy()

    print("Primeras filas del dataset:")
    print(data[:5])

    # Calcular retornos
    returns_matrix = data[1:] / data[:-1] - 1

    return returns_matrix