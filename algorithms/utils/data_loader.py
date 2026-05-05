import pandas as pd
import numpy as np

# This module contains functions to load datasets for portfolio optimization.

def load_historical_returns(file_path, frequency='daily'):
    """
    Load historical price/return data and return the returns matrix for MDD calculation.
    
    Parameters:
    - file_path: path to CSV file with historical data
    - frequency: 'daily', 'weekly', 'monthly' (for informational purposes)
    
    Returns:
    - returns_matrix: numpy array of shape (n_assets, n_periods) with historical returns
    - asset_names: list of asset names
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # If data are prices, convert to returns
    if df.iloc[0].mean() > 0.1:  # Probably prices
        returns_df = df.pct_change().dropna()
    else:  # Probably already returns
        returns_df = df.copy()
    
    # Remove columns with too many NaN or invalid values
    returns_df = returns_df.dropna(axis=1, thresh=len(returns_df)*0.5)
    returns_df = returns_df.fillna(0)
    
    # Remove columns with -99.99 or -999 codes
    valid_columns = ~returns_df.isin([-99.99, -999]).any()
    returns_df = returns_df.loc[:, valid_columns]
    
    # Transpose to get (n_assets, n_periods)
    returns_matrix = returns_df.values.T
    
    print(f"Historical returns matrix shape: {returns_matrix.shape}")
    print(f"Number of assets: {returns_matrix.shape[0]}")
    print(f"Number of periods: {returns_matrix.shape[1]}")
    
    return returns_matrix, returns_df.columns.tolist()


def load_data_for_both_models(file_path):
    """
    Load data and return both formats for MDD and Markowitz approaches.
    Useful for comparing both methods in your TFG.
    """
    returns_matrix, asset_names = load_historical_returns(file_path)
    
    mean_returns = np.mean(returns_matrix, axis=1)
    cov_matrix = np.cov(returns_matrix)
    
    return {
        'returns_matrix': returns_matrix,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'asset_names': asset_names
    }


# Keep original functions for backward compatibility
def load_dataset(file_path):
    """Original function for Markowitz model with synthetic data."""
    raw_lines = pd.read_csv(file_path, header=None, dtype=str).squeeze().tolist()
    num_assets = int(raw_lines[0])
    
    stats_df = pd.read_csv(
        file_path, 
        sep=r'\s+', 
        skiprows=1, 
        nrows=num_assets, 
        header=None, 
        names=['return', 'std_dev']
    ).astype(float)

    corr_matrix = pd.DataFrame(1.0, index=range(num_assets), columns=range(num_assets))
    
    correlations = pd.read_csv(
        file_path,
        sep=r'\s+',
        skiprows=num_assets + 1,
        header=None,
        names=['i', 'j', 'corr']
    )

    for _, row in correlations.iterrows():
        i_idx, j_idx = int(row['i']) - 1, int(row['j']) - 1
        corr = float(row['corr'])
        corr_matrix.iat[i_idx, j_idx] = corr
        corr_matrix.iat[j_idx, i_idx] = corr

    std_devs = stats_df['std_dev']
    std_outer = std_devs.to_frame().dot(std_devs.to_frame().T)
    cov_matrix = std_outer * corr_matrix

    return stats_df['return'].values, cov_matrix.values


def load_real_data(file_path):
    """Original function for Markowitz model with real data."""
    df = pd.read_csv(file_path, header=None, skiprows=1, dtype=str)
    df = df.drop(columns=[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    columnas_validas = df.columns[~df.isin([-99.99, -999]).any()]
    df_limpio = df[columnas_validas]
    
    retuns = df_limpio.mean().to_numpy()
    cov_matrix = df_limpio.cov().to_numpy() / 100
    
    return retuns, cov_matrix


def save_data_txt(file_path, output_path):
    """Utility to save data in text format."""
    returns, cov_matrix = load_real_data(file_path)
    N = len(returns)
    stds = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(stds, stds)
    
    with open(output_path, 'w') as f:
        f.write(f"{N}\n")
        for i in range(N):
            f.write(f"{returns[i]:.6f} {stds[i]:.6f}\n")
        for i in range(N):
            for j in range(i, N):
                f.write(f"{i+1} {j+1} {corr_matrix[i, j]:.6f}\n")
    
    print(f"File saved in: {output_path}")