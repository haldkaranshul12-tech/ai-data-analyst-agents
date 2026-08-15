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