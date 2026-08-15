import streamlit as st
from src.ingestion.loader import load_dataset, get_dataset_info

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

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload a CSV or Excel file to get started.")