"""
Module 5: AI Data Analyst Agent - Core
Connects the LLM (Groq) to the analysis tools using function/tool calling.
The AI reads the user's question, picks the right tool, runs it on the
dataset, and explains the result in plain English.
"""
import os
import json
import re


def _extract_failed_tool_call(error_text):
    """
    Groq's model sometimes wraps a tool call in <function=...></function>
    tags instead of using proper tool calling format. This extracts the
    function name and arguments from that malformed text as a fallback.
    """
    match = re.search(r"<function=(\w+)(\{.*?\})", error_text)
    if not match:
        return None, None
    function_name = match.group(1)
    try:
        function_args = json.loads(match.group(2))
    except json.JSONDecodeError:
        return None, None
    return function_name, function_args
from groq import Groq
from dotenv import load_dotenv
from src.agents.ai_agents import AVAILABLE_TOOLS

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_tool_definitions(column_names):
    """
    Describes each available tool to the LLM in the format it expects
    for tool/function calling.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_column_average",
                "description": "Get the average (mean) of a numeric column",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string", "enum": column_names}},
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_column_sum",
                "description": "Get the sum of a numeric column",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string", "enum": column_names}},
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_column_max",
                "description": "Get the maximum value of a column",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string", "enum": column_names}},
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_column_min",
                "description": "Get the minimum value of a column",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string", "enum": column_names}},
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_column_median",
                "description": "Get the median of a numeric column",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string", "enum": column_names}},
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_value_counts",
                "description": "Get the count of each unique value in a column",
                "parameters": {
                    "type": "object",
                    "properties": {"column": {"type": "string", "enum": column_names}},
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filter_and_count",
                "description": "Count rows where a column equals a specific value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "enum": column_names},
                        "value": {"type": "string"},
                    },
                    "required": ["column", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_row_count",
                "description": "Get the total number of rows in the dataset",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_correlation_between",
                "description": "Get the correlation coefficient between two numeric columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col1": {"type": "string", "enum": column_names},
                        "col2": {"type": "string", "enum": column_names},
                    },
                    "required": ["col1", "col2"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "group_and_aggregate",
                "description": "Group by a column and aggregate another numeric column (mean, sum, count, min, max)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_col": {"type": "string", "enum": column_names},
                        "agg_col": {"type": "string", "enum": column_names},
                        "agg_func": {"type": "string", "enum": ["mean", "sum", "count", "min", "max"]},
                    },
                    "required": ["group_col", "agg_col", "agg_func"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filter_and_average",
                "description": "Filter rows where a column equals a value, then average another numeric column for those rows",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter_col": {"type": "string", "enum": column_names},
                        "filter_value": {"type": "string"},
                        "avg_col": {"type": "string", "enum": column_names},
                    },
                    "required": ["filter_col", "filter_value", "avg_col"],
                },
            },
        },
    ]


def ask_data_analyst(user_question, df, chat_history=None, retry=True):
    """
    Main agent function: takes a natural language question and the dataset,
    lets the LLM pick and call the right tool, runs it, and returns a
    plain-English answer. Uses chat_history for follow-up context.

    Parameters
    ----------
    chat_history : list of (question, answer) tuples from earlier in the conversation
    """
    column_names = df.columns.tolist()
    tools = build_tool_definitions(column_names)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful data analyst. The dataset has these columns: {column_names}. "
                "You have access to tools that can compute statistics on this dataset. "
                "Use the appropriate tool to answer the user's question. "
                "IMPORTANT: Only answer questions that relate to this dataset and its columns. "
                "If the user asks something unrelated to the dataset (general knowledge, "
                "opinions, jokes, coding help, or anything not about this data), "
                "politely respond that you can only answer questions about the uploaded dataset, "
                "and do NOT call any tool in that case. "
                "Use the conversation history to understand follow-up questions "
                "(e.g. 'what about for females?' refers back to the previous question's topic)."
            ),
        },
    ]

    # Add prior conversation turns for context (limit to last 5 to keep it light)
    if chat_history:
        for past_q, past_a in chat_history[-5:]:
            messages.append({"role": "user", "content": past_q})
            messages.append({"role": "assistant", "content": past_a})

    messages.append({"role": "user", "content": user_question})

    function_name = None
    function_args = None
    used_fallback = False

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        response_message = response.choices[0].message

        if not response_message.tool_calls:
            return response_message.content or "I couldn't determine how to answer that."

        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

    except Exception as e:
        function_name, function_args = _extract_failed_tool_call(str(e))
        if function_name is None:
            return "I had trouble understanding that question. Could you try rephrasing it more simply?"
        used_fallback = True

    if function_name not in AVAILABLE_TOOLS:
        return f"Sorry, I don't have a tool called '{function_name}'."

    tool_function = AVAILABLE_TOOLS[function_name]

    try:
        if function_args:
            result = tool_function(df, **function_args)
        else:
            result = tool_function(df)
    except Exception as e:
        return f"I tried to analyze this but ran into an error: {e}"
    if used_fallback:
        return f"Here's what I found: {result}"

    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": json.dumps(result, default=str),
    })

    try:
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
        )
        return final_response.choices[0].message.content
    except Exception:
        return f"Here's what I found: {result}"