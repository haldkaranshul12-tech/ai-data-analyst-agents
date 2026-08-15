"""
Module 3: Exploratory Data Analysis (EDA)
Provides descriptive statistics, correlation, and group-wise analysis.
"""
import pandas as pd


def get_descriptive_stats(df):
    """
    Returns descriptive statistics for numeric columns.
    """
    return df.describe()


def get_categorical_summary(df):
    """
    Returns a summary of categorical (non-numeric) columns:
    unique value count and most frequent value.
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


def get_correlation_matrix(df):
    """
    Returns the correlation matrix for numeric columns only.
    """
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr()


def get_group_summary(df, group_col, agg_col, agg_func="mean"):
    """
    Groups the dataframe by group_col and aggregates agg_col.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str - column to group by (e.g. "Sex")
    agg_col : str - numeric column to aggregate (e.g. "Age")
    agg_func : str - "mean", "sum", "count", "min", "max"

    Returns
    -------
    pd.DataFrame with grouped results
    """
    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_col, f"{agg_func}_{agg_col}"]
    return result