import streamlit as st
from src.ingestion.loader import load_dataset, get_dataset_info
from src.profiling.cleaner import (
    profile_dataset,
    find_duplicates,
    remove_duplicates,
    handle_missing_values,
)

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("🤖 AI Data Analyst Agent")
st.caption("Upload a dataset, explore it, and ask questions in plain English.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")

        info = get_dataset_info(df)

        col1, col2 = st.columns(2)
        col1.metric("Rows", info["rows"])
        col2.metric("Columns", info["columns"])

        st.subheader("Column Details")
        st.write(info["dtypes"])

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Data Profiling")
        profile_df = profile_dataset(df)
        st.dataframe(profile_df)

        st.subheader("Data Cleaning")

        dup_count = find_duplicates(df)
        st.write(f"Duplicate rows found: *{dup_count}*")

        if st.button("Remove Duplicates"):
            df = remove_duplicates(df)
            st.success(f"Duplicates removed. New shape: {df.shape}")
            st.dataframe(df.head())

        st.write("Handle Missing Values")
        strategy = st.radio("Choose a strategy:", ["drop", "fill"])

        if strategy == "fill":
            fill_value = st.text_input("Value to fill missing cells with:", "0")
        else:
            fill_value = None

        if st.button("Apply Missing Value Handling"):
            cleaned_df = handle_missing_values(df, strategy=strategy, fill_value=fill_value)
            st.success(f"Missing values handled. New shape: {cleaned_df.shape}")
            st.dataframe(cleaned_df.head())

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.info("upload a CSV or Excel file to get started.")