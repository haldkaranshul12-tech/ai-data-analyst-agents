"""
Module 1: Data Ingestion
Handles reading CSV and Excel files into a pandas DataFrame.
"""
import pandas as pd


def load_dataset(uploaded_file):
    """
    Reads an uploaded CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : file-like object (e.g. Streamlit's UploadedFile)

    Returns
    -------
    pd.DataFrame
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    return df
def get_dataset_info(df):
    """
    Returns basic structural info about the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with rows, columns, column names, and column data types
    """
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return info