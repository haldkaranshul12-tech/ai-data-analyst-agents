"""
Module 4: LLM-based Visualization Agent
Uses Groq API to interpret a natural language request and decide
which chart type and columns to use.
"""
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def interpret_chart_request(user_query, column_names):
    """
    Sends the user's natural language request to Groq and asks it to
    pick a chart type and relevant column(s) from the dataset.

    Parameters
    ----------
    user_query : str - e.g. "show me age distribution"
    column_names : list of str - available columns in the dataset

    Returns
    -------
    dict like:
        {
            "chart_type": "histogram" | "bar" | "line" | "scatter" | "heatmap",
            "col1": "<column name or null>",
            "col2": "<column name or null>"
        }
    """
    system_prompt = f"""You are a data visualization assistant.
The dataset has these columns: {column_names}

Given a user's request, respond ONLY with a JSON object (no extra text, no markdown)
in this exact format:
{{"chart_type": "histogram|bar|line|scatter|heatmap", "col1": "<column name>", "col2": "<column name or null>"}}

Rules:
- chart_type must be one of: histogram, bar, line, scatter, heatmap
- col1 and col2 must be exact column names from the list above, or null
- histogram needs only col1
- heatmap needs neither col1 nor col2 (use null, null)
- bar/line/scatter typically need both col1 and col2
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case the model wraps it in markdown fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"chart_type": "bar", "col1": column_names[0], "col2": None}

    return result