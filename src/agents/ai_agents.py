"""
Module 5: AI Data Analyst Agent
Core analysis tools that the AI agent can call to answer
natural language questions about the dataset.
"""
import pandas as pd


def get_column_average(df, column):
    """Returns the mean of a numeric column."""
    return round(df[column].mean(), 2)


def get_column_sum(df, column):
    """Returns the sum of a numeric column."""
    return round(df[column].sum(), 2)


def get_column_max(df, column):
    """Returns the max value of a column."""
    return df[column].max()


def get_column_min(df, column):
    """Returns the min value of a column."""
    return df[column].min()


def get_value_counts(df, column):
    """Returns the count of each unique value in a column."""
    return df[column].value_counts().to_dict()


def filter_and_count(df, column, value):
    """Counts rows where column equals a given value."""
    return int((df[column] == value).sum())


def get_row_count(df):
    """Returns the total number of rows in the dataset."""
    return len(df)


def get_column_median(df, column):
    """Returns the median of a numeric column."""
    return round(df[column].median(), 2)


def get_correlation_between(df, col1, col2):
    """Returns the correlation coefficient between two numeric columns."""
    return round(df[col1].corr(df[col2]), 3)


def group_and_aggregate(df, group_col, agg_col, agg_func="mean"):
    """
    Groups by group_col and aggregates agg_col using agg_func.
    agg_func can be: mean, sum, count, min, max
    """
    result = df.groupby(group_col)[agg_col].agg(agg_func)
    return result.to_dict()


def filter_and_average(df, filter_col, filter_value, avg_col):
    """
    Filters rows where filter_col equals filter_value,
    then returns the average of avg_col for those rows.
    """
    filtered = df[df[filter_col] == filter_value]
    if len(filtered) == 0:
        return None
    return round(filtered[avg_col].mean(), 2)


# Registry of available tools — used by the agent to know what it can call
AVAILABLE_TOOLS = {
    "get_column_average": get_column_average,
    "get_column_sum": get_column_sum,
    "get_column_max": get_column_max,
    "get_column_min": get_column_min,
    "get_column_median": get_column_median,
    "get_value_counts": get_value_counts,
    "filter_and_count": filter_and_count,
    "get_row_count": get_row_count,
    "get_correlation_between": get_correlation_between,
    "group_and_aggregate": group_and_aggregate,
    "filter_and_average": filter_and_average,
}