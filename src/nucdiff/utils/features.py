import pandas as pd
import numpy as np
from .nuclear_data import MAGIC_NUMBERS

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of engineered physics features to the DataFrame.
    """
    df['n'] = df['A'] - df['Z']

    # --- Odd/Even Features ---
    df['is_odd_Z'] = (df['Z'] % 2).astype(int)
    df['is_odd_N'] = (df['n'] % 2).astype(int)

    # --- Magic Number and Valence Features ---
    magic_numbers_arr = np.array(list(MAGIC_NUMBERS))
    
    def get_delta_magic(nucleon_number):
        return np.min(np.abs(nucleon_number - magic_numbers_arr))
        
    def get_valence(nucleon_number):
        # Find the largest magic number smaller than or equal to the nucleon number
        shell_base = magic_numbers_arr[magic_numbers_arr <= nucleon_number]
        if len(shell_base) > 0:
            return nucleon_number - shell_base.max()
        return nucleon_number # For very light nuclei

    df['delta_magic_z'] = df['Z'].apply(get_delta_magic)
    df['delta_magic_n'] = df['n'].apply(get_delta_magic)
    df['valence_p'] = df['Z'].apply(get_valence)
    df['valence_n'] = df['n'].apply(get_valence)
    
    # --- Asymmetry Feature ---
    df['asym'] = (df['n'] - df['Z']) / (df['A'] + 1e-6)
    
    return df 