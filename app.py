import streamlit as st
from src.ingestion.loader import load_dataset, get_dataset_info
from src.profiling.cleaner import (
    profile_dataset,
    find_duplicates,
    remove_duplicates,
    handle_missing_values,
)
from src.eda.analysis import (
    get_descriptive_stats,
    get_categorical_summary,
    get_correlation_matrix,
    get_group_summary,
)

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("🤖 AI Data Analyst Agent")
st.caption("Upload a dataset, explore it, and ask questions in plain English.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
            st.session_state["df"] = load_dataset(uploaded_file)
            st.session_state["filename"] = uploaded_file.name

        df = st.session_state["df"]
        st.success(f"Loaded: {uploaded_file.name}")

        info = get_dataset_info(df)
        col1, col2 = st.columns(2)
        col1.metric("Rows", info["rows"])
        col2.metric("Columns", info["columns"])

        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview & Cleaning",
            "Descriptive Stats",
            "Correlation",
            "Group Analysis",
        ])

        with tab1:
            st.subheader("Column Details")
            st.write(info["dtypes"])

            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("Data Profiling")
            profile_df = profile_dataset(df)
            st.dataframe(profile_df)

            st.subheader("Data Cleaning")
            dup_count = find_duplicates(df)
            st.write(f"Duplicate rows found: {dup_count}")

            if st.button("Remove Duplicates"):
                st.session_state["df"] = remove_duplicates(df)
                st.success(f"Duplicates removed. New shape: {st.session_state['df'].shape}")
                st.rerun()

            st.write("Handle Missing Values")
            strategy = st.radio("Choose a strategy:", ["drop", "fill"])
            if strategy == "fill":
                fill_value = st.text_input("Value to fill missing cells with:", "0")
            else:
                fill_value = None

            if st.button("Apply Missing Value Handling"):
                st.session_state["df"] = handle_missing_values(df, strategy=strategy, fill_value=fill_value)
                st.success(f"Missing values handled. New shape: {st.session_state['df'].shape}")
                st.rerun()

            st.divider()
            st.subheader("Current Cleaned Dataset")
            st.dataframe(st.session_state["df"].head())
            st.caption(f"Shape: {st.session_state['df'].shape[0]} rows x {st.session_state['df'].shape[1]} columns")

        with tab2:
            st.subheader("Descriptive Statistics (Numeric Columns)")
            st.dataframe(get_descriptive_stats(st.session_state["df"]))

            st.subheader("Categorical Columns Summary")
            cat_summary = get_categorical_summary(st.session_state["df"])
            if not cat_summary.empty:
                st.dataframe(cat_summary)
            else:
                st.write("No categorical columns found.")

        with tab3:
            st.subheader("Correlation Matrix")
            corr_matrix = get_correlation_matrix(st.session_state["df"])
            st.dataframe(corr_matrix)

        with tab4:
            st.subheader("Group-wise Analysis")
            numeric_cols = st.session_state["df"].select_dtypes(include="number").columns.tolist()
            all_cols = st.session_state["df"].columns.tolist()

            gcol1, gcol2, gcol3 = st.columns(3)
            group_col = gcol1.selectbox("Group by column:", all_cols)
            agg_col = gcol2.selectbox("Aggregate column (numeric):", numeric_cols)
            agg_func = gcol3.selectbox("Aggregation:", ["mean", "sum", "count", "min", "max"])

            if st.button("Run Group Analysis"):
                group_result = get_group_summary(st.session_state["df"], group_col, agg_col, agg_func)
                st.dataframe(group_result)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.session_state.clear()
    st.info("Upload a CSV or Excel file to get started.")