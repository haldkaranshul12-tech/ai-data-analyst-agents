"""
Module 4: Visualization
Generates interactive Plotly charts for the dataset.
"""
import pandas as pd
import plotly.express as px


def bar_chart(df, x_col, y_col):
    """Creates a bar chart."""
    return px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")


def line_chart(df, x_col, y_col):
    """Creates a line chart."""
    return px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")


def histogram_chart(df, col):
    """Creates a histogram for a numeric column."""
    return px.histogram(df, x=col, title=f"Distribution of {col}")


def scatter_chart(df, x_col, y_col, color_col=None):
    """Creates a scatter plot, optionally colored by a category."""
    return px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")


def heatmap_chart(corr_matrix):
    """Creates a heatmap from a correlation matrix."""
    return px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
    )


def auto_select_chart(df, col1, col2=None):
    """
    Automatically picks the best chart type based on column data types.

    Rules:
    - One numeric column only -> Histogram
    - One categorical column only -> Bar chart (value counts)
    - Two numeric columns -> Scatter plot
    - One numeric + one categorical -> Bar chart (grouped mean)
    - Two categorical columns -> Bar chart (count by group)

    Returns
    -------
    (plotly figure, chart_type_used)
    """
    is_col1_numeric = pd.api.types.is_numeric_dtype(df[col1])

    # Only one column selected
    if col2 is None:
        if is_col1_numeric:
            return histogram_chart(df, col1), "Histogram"
        else:
            counts = df[col1].value_counts().reset_index()
            counts.columns = [col1, "count"]
            return bar_chart(counts, col1, "count"), "Bar Chart"

    is_col2_numeric = pd.api.types.is_numeric_dtype(df[col2])

    # Two numeric columns -> scatter
    if is_col1_numeric and is_col2_numeric:
        return scatter_chart(df, col1, col2), "Scatter Plot"

    # One numeric + one categorical -> bar chart of mean
    if is_col1_numeric and not is_col2_numeric:
        grouped = df.groupby(col2)[col1].mean().reset_index()
        return bar_chart(grouped, col2, col1), "Bar Chart (avg)"

    if is_col2_numeric and not is_col1_numeric:
        grouped = df.groupby(col1)[col2].mean().reset_index()
        return bar_chart(grouped, col1, col2), "Bar Chart (avg)"

    # Both categorical -> count chart
    counts = df.groupby([col1, col2]).size().reset_index(name="count")
    return bar_chart(counts, col1, "count"), "Bar Chart (count)"