"""
Module 6: Insights & Report Generation
Automatically generates key insights, recommendations, and PDF reports.
"""
import os
import io
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_dataset_summary(df):
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
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a skilled data analyst who writes clear, concise insights."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def generate_recommendations(df):
    dataset_summary = build_dataset_summary(df)

    prompt = f"""Based on the following dataset summary, generate 4-6 practical
recommendations for someone working with this data. Focus on data quality
improvements (handling missing values, duplicates), potential feature
engineering ideas, and suggestions for further analysis. Write each
recommendation as a short, actionable bullet point in plain English.

Dataset summary:
{dataset_summary}

Respond with only the bullet points, one per line, starting each with "- ".
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a skilled data analyst who gives practical, actionable recommendations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def generate_pdf_report(df, filename_label, insights_text, recommendations_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20)
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    elements.append(Paragraph("AI Data Analyst Report", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Dataset: {filename_label}", normal_style))
    elements.append(Paragraph(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}", normal_style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Key Insights", heading_style))
    elements.append(Spacer(1, 8))
    if insights_text:
        for line in insights_text.split("\n"):
            line = line.strip().lstrip("- ").strip()
            if line:
                elements.append(Paragraph(f"&bull; {line}", normal_style))
                elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("No insights generated yet.", normal_style))

    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Recommendations", heading_style))
    elements.append(Spacer(1, 8))
    if recommendations_text:
        for line in recommendations_text.split("\n"):
            line = line.strip().lstrip("- ").strip()
            if line:
                elements.append(Paragraph(f"&bull; {line}", normal_style))
                elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("No recommendations generated yet.", normal_style))

    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Column Overview", heading_style))
    elements.append(Spacer(1, 8))

    table_data = [["Column", "Type", "Missing"]]
    missing_counts = df.isnull().sum()
    for col in df.columns:
        table_data.append([str(col), str(df[col].dtype), str(missing_counts[col])])

    table = Table(table_data, colWidths=[200, 100, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer
