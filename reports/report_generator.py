"""
Module 6: Insights & Report Generation
Automatically generates key insights and recommendations from the dataset.
"""
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_dataset_summary(df):
    """
    Builds a compact text summary of the dataset (stats, missing values,
    correlations) that can be fed to the LLM to generate insights.
    """
    summary_parts = []

    summary_parts.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        summary_parts.append("\nNumeric column statistics:")
        summary_parts.append(numeric_df.describe().round(2).to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        summary_parts.append("\nMissing values per column:")
        summary_parts.append(missing.to_string())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        summary_parts.append("\nCategorical column top values:")
        for col in cat_cols[:5]:
            top_values = df[col].value_counts().head(3)
            summary_parts.append(f"{col}: {top_values.to_dict()}")

    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        summary_parts.append("\nCorrelation matrix (numeric columns):")
        summary_parts.append(corr.round(2).to_string())

    return "\n".join(summary_parts)


def generate_key_insights(df):
    """
    Sends a dataset summary to the LLM and asks it to generate
    key insights in plain English, as a bulleted list.
    """
    dataset_summary = build_dataset_summary(df)

    prompt = f"""Based on the following dataset summary, generate 5-7 key insights
about this data. Focus on interesting patterns, notable statistics, data quality
issues (like missing values), and relationships between columns. Write each
insight as a short, clear bullet point in plain English. Do not make up
information that isn't supported by the summary below.

Dataset summary:
{dataset_summary}

Respond with only the bullet points, one per line, starting each with "- ".
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",        messages=[
            {"role": "system", "content": "You are a skilled data analyst who writes clear, concise insights."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content