"""
Module 4: Visualization
Generates interactive Plotly charts for the dataset.
"""
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