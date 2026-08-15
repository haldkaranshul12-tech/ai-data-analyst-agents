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
"""
Module 2: Data Profiling & Cleaning
Generates a profile summary of the dataset, detects duplicates,
and handles missing values.
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


def find_duplicates(df):
    """
    Returns the number of duplicate rows in the dataset.
    """
    return df.duplicated().sum()


def remove_duplicates(df):
    """
    Returns a copy of the dataset with duplicate rows removed.
    """
    return df.drop_duplicates()


def handle_missing_values(df, strategy="drop", fill_value=None):
    """
    Handles missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        "drop" - removes rows with any missing values
        "fill" - fills missing values with fill_value
    fill_value : any
        Value to use when strategy is "fill"

    Returns
    -------
    pd.DataFrame with missing values handled
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill":
        return df.fillna(fill_value)
    else:
        raise ValueError("strategy must be 'drop' or 'fill'")