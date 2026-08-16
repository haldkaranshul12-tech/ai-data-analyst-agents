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
from src.visualization.charts import (
    bar_chart,
    line_chart,
    histogram_chart,
    scatter_chart,
    heatmap_chart,
    auto_select_chart,
)
from src.visualization.llm_viz_agent import interpret_chart_request
from src.agents.agent_core import ask_data_analyst

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("AI Data Analyst Agent")
st.caption("Upload a dataset, explore it, and ask questions in plain English.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
            st.session_state["df"] = load_dataset(uploaded_file)
            st.session_state["filename"] = uploaded_file.name
            st.session_state["chat_history"] = []

        df = st.session_state["df"]
        st.success(f"Loaded: {uploaded_file.name}")

        info = get_dataset_info(df)
        col1, col2 = st.columns(2)
        col1.metric("Rows", info["rows"])
        col2.metric("Columns", info["columns"])

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview & Cleaning",
            "Descriptive Stats",
            "Correlation",
            "Group Analysis",
            "Charts",
            "Ask Questions",
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

        with tab5:
            current_df = st.session_state["df"]
            numeric_cols = current_df.select_dtypes(include="number").columns.tolist()
            all_cols = current_df.columns.tolist()

            st.subheader("Ask AI to Create a Chart")
            user_query = st.text_input(
                "Describe the chart you want:",
                placeholder="e.g. show me age distribution",
            )

            if st.button("Ask AI"):
                if user_query.strip() == "":
                    st.warning("Please type a request first.")
                else:
                    with st.spinner("Thinking..."):
                        result = interpret_chart_request(user_query, all_cols)

                    chart_type = result.get("chart_type")
                    c1 = result.get("col1")
                    c2 = result.get("col2")
                    st.caption(f"AI chose: {chart_type} | col1={c1} | col2={c2}")

                    try:
                        if chart_type == "histogram":
                            st.plotly_chart(histogram_chart(current_df, c1))
                        elif chart_type == "bar":
                            st.plotly_chart(bar_chart(current_df, c1, c2))
                        elif chart_type == "line":
                            st.plotly_chart(line_chart(current_df, c1, c2))
                        elif chart_type == "scatter":
                            st.plotly_chart(scatter_chart(current_df, c1, c2))
                        elif chart_type == "heatmap":
                            corr = get_correlation_matrix(current_df)
                            st.plotly_chart(heatmap_chart(corr))
                        else:
                            st.error("Could not determine a suitable chart.")
                    except Exception as chart_error:
                        st.error(f"Could not create chart: {chart_error}")

            st.divider()
            st.subheader("Auto Chart (recommended for you)")
            acol1, acol2 = st.columns(2)
            auto_col1 = acol1.selectbox("Column 1:", all_cols, key="auto_col1")
            auto_col2 = acol2.selectbox("Column 2 (optional):", [None] + all_cols, key="auto_col2")

            if st.button("Auto-Generate Chart"):
                fig, chart_used = auto_select_chart(current_df, auto_col1, auto_col2)
                st.caption(f"Auto-selected: {chart_used}")
                st.plotly_chart(fig)

            st.divider()
            st.subheader("Manual Charts")

            chart_type_manual = st.selectbox(
                "Choose chart type:",
                ["Bar", "Line", "Histogram", "Scatter", "Heatmap"],
            )

            if chart_type_manual == "Bar":
                x_col = st.selectbox("X-axis:", all_cols, key="bar_x")
                y_col = st.selectbox("Y-axis:", numeric_cols, key="bar_y")
                if st.button("Generate Bar Chart"):
                    st.plotly_chart(bar_chart(current_df, x_col, y_col))

            elif chart_type_manual == "Line":
                x_col = st.selectbox("X-axis:", all_cols, key="line_x")
                y_col = st.selectbox("Y-axis:", numeric_cols, key="line_y")
                if st.button("Generate Line Chart"):
                    st.plotly_chart(line_chart(current_df, x_col, y_col))

            elif chart_type_manual == "Histogram":
                col = st.selectbox("Column:", numeric_cols, key="hist_col")
                if st.button("Generate Histogram"):
                    st.plotly_chart(histogram_chart(current_df, col))

            elif chart_type_manual == "Scatter":
                x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Color by (optional):", [None] + all_cols, key="scatter_color")
                if st.button("Generate Scatter Plot"):
                    st.plotly_chart(scatter_chart(current_df, x_col, y_col, color_col))

            elif chart_type_manual == "Heatmap":
                if st.button("Generate Heatmap"):
                    corr = get_correlation_matrix(current_df)
                    st.plotly_chart(heatmap_chart(corr))

        with tab6:
            st.subheader("Ask Questions About Your Data")
            st.caption("Example: What's the average age? How many people survived? What's the most common embarkation point?")

            question = st.text_input("Type your question:", key="qa_input")

            if st.button("Get Answer"):
                if question.strip() == "":
                    st.warning("Please type a question first.")
                else:
                    with st.spinner("Analyzing..."):
                        answer = ask_data_analyst(
                            question,
                            st.session_state["df"],
                            chat_history=st.session_state["chat_history"],
                        )
                    st.session_state["chat_history"].append((question, answer))

            if st.session_state.get("chat_history"):
                st.divider()
                st.subheader("Conversation History")
                for q, a in reversed(st.session_state["chat_history"]):
                    st.markdown(f"**Q: {q}**")
                    st.write(a)
                    st.markdown("---")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.session_state.clear()
    st.info("Upload a CSV or Excel file to get started.")