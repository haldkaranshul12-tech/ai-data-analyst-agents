"""
Module 2: Data Profiling
Generates a profile summary of the dataset — missing values,
unique values, and data types per column.
"""
import pandas as pd


def profile_dataset(df):
    """
    Builds a per-column profile summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with one row per column, showing:
        - dtype
        - missing_count
        - missing_percent
        - unique_count
    """
    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isnull().sum().values,
        "missing_percent": (df.isnull().sum().values / len(df) * 100).round(2),
        "unique_count": df.nunique().values,
    })
    return profile