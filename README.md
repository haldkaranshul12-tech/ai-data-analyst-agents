# 🤖 AI Data Analyst Agent

An AI-powered data analysis tool that lets you upload a dataset, explore
it visually, and ask questions about it in plain English — powered by
Groq's LLM API for natural language understanding and automated insights.

## Features

- **Data Ingestion** — Upload CSV or Excel files
- **Data Profiling & Cleaning** — Missing values, duplicates, data types
- **Exploratory Data Analysis** — Descriptive stats, correlations, group analysis
- **Visualization** — Bar, line, histogram, scatter, and heatmap charts, plus
  an AI agent that builds the right chart from a plain-English request
- **AI Data Analyst Agent** — Ask questions about your dataset in natural
  language; the agent picks the right analysis tool and answers conversationally
- **Automatic Insights & Recommendations** — AI-generated key findings and
  practical suggestions
- **PDF Report Export** — Download a full report with insights, recommendations,
  and a column summary table

## Tech Stack

| Layer | Tool |
|---|---|
| Core | Python |
| Data Processing | Pandas + NumPy |
| Visualization | Plotly |
| AI / LLM | Groq API (Llama models via OpenAI-compatible chat completions) |
| User Interface | Streamlit |
| PDF Report Generation | ReportLab |
| Version Control | Git / GitHub |

## Project Structure
ai-data-analyst-agent/
├── data/ # sample datasets (gitignored)
├── src/
│ ├── ingestion/ # file upload and reading
│ ├── profiling/ # data profiling and cleaning
│ ├── eda/ # descriptive stats, correlation, grouping
│ ├── visualization/ # charts + AI chart selection
│ ├── agents/ # AI analyst agent (tool calling)
│ └── reports/ # insights, recommendations, PDF export
├── app.py # Streamlit entry point
├── requirements.txt
└── .env # GROQ_API_KEY (not committed)


## Setup

1. Clone the repository
2. Create a virtual environment and activate it:
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Add your Groq API key to a `.env` file:

GROQ_API_KEY=your_key_here
5. Run the app:
```bash
   streamlit run app.py
```

## How to Use

1. Upload a CSV or Excel file
2. Explore the **Overview & Cleaning** tab to profile and clean your data
3. Check **Descriptive Stats**, **Correlation**, and **Group Analysis** for EDA
4. Use the **Charts** tab to visualize data — manually, automatically, or by
   asking the AI in plain English
5. Ask any question about your data in the **Ask Questions** tab
6. Generate **Key Insights** and **Recommendations**, then download a full
   **PDF report**

## Project Status

Built as a 30-day minor project. All 4 core modules are complete:

- ✅ Module 1 — Data Ingestion & Profiling
- ✅ Module 2 — EDA & Visualization
- ✅ Module 3 — AI Analyst Agent
- ✅ Module 4 — Insights & Reports