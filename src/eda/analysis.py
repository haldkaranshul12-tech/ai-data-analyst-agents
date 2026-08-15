"""
Module 3: Exploratory Data Analysis (EDA)
Provides descriptive statistics for numeric and categorical columns.
"""
import pandas as pd


def get_descriptive_stats(df):
    """
    Returns descriptive statistics for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with count, mean, std, min, max, quartiles
    """
    return df.describe()


def get_categorical_summary(df):
    """
    Returns a summary of categorical (non-numeric) columns:
    unique value count and most frequent value.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with column, unique_count, most_common, most_common_count
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    summary = []
    for col in cat_cols:
        value_counts = df[col].value_counts()
        summary.append({
            "column": col,
            "unique_count": df[col].nunique(),
            "most_common": value_counts.index[0] if not value_counts.empty else None,
            "most_common_count": value_counts.iloc[0] if not value_counts.empty else 0,
        })

    return pd.DataFrame(summary)