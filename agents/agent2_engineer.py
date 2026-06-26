

import json, os, sys
import pandas as pd
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.agent_tools import (
    inspect_metadata, get_column_stats,
    create_interaction, encode_categorical,
    correlation_analysis, select_top_features,
)

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_metadata",
            "description": "Inspect dataset shape, dtypes, and null counts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_stats",
            "description": "Get detailed statistics for one column.",
            "parameters": {
                "type": "object",
                "properties": {"col": {"type": "string"}},
                "required": ["col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_interaction",
            "description": (
                "Create a new feature from a math expression using existing columns. "
                "Use pandas eval syntax, e.g. expression='Fare / (SibSp + Parch + 1)'. "
                "Give the new column a descriptive snake_case name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression":   {"type": "string", "description": "Math expression using column names"},
                    "new_col_name": {"type": "string", "description": "Name for the new column"},
                },
                "required": ["expression", "new_col_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "encode_categorical",
            "description": (
                "Encode a string/categorical column to numbers. "
                "Use method='onehot' for low-cardinality (<10 unique), method='label' otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col":    {"type": "string"},
                    "method": {"type": "string", "enum": ["onehot", "label"]},
                },
                "required": ["col", "method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_analysis",
            "description": "Compute Pearson correlations of all numeric features with the target column.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_top_features",
            "description": "Keep only the k most correlated features plus the target. Call this last.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "k":      {"type": "integer", "description": "Number of features to keep (5-10)"},
                },
                "required": ["target", "k"],
            },
        },
    },
]


def run_agent2(clean_csv_path, agent1_summary, target_column, output_dir, log_lines):
    client = OpenAI()
    df = pd.read_csv(clean_csv_path)

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("\n" + "="*60)
    log("AGENT 2 — THE FEATURE ENGINEER  ('The Architect')")
    log("="*60)
    log(f"Loaded: {clean_csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    system_prompt = f"""You are an expert Feature Engineering Agent.
You receive a cleaned dataset. Your goal is to maximize predictive signal for: '{target_column}'.

Rules:
- Start with inspect_metadata.
- Encode ALL remaining string/categorical columns before running correlation_analysis.
- Create at least ONE meaningful ratio or interaction feature (e.g. fare per family member).
- Run correlation_analysis after encoding to see what matters.
- Use select_top_features to keep only 5-8 of the best features.
- Explain your reasoning for every feature you create or drop.
- When done, write a summary in EXACTLY this format:

ENGINEERING SUMMARY:
- <action> | Reason: <why>
- ...
FINAL FEATURES: <comma-separated list>"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"Data Cleaner summary:\n{agent1_summary}\n\n"
            f"Now engineer features for predicting '{target_column}'. "
            f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols."
        )},
    ]

    summary = ""

    for _ in range(25):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            log(f"\n[Agent 2 thinks]: {msg.content}")
            if "ENGINEERING SUMMARY:" in msg.content:
                summary = msg.content

        if not msg.tool_calls:
            log("\n[Agent 2]: Finished feature engineering.")
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            log(f"\n[Agent 2 calls tool]: {name}({json.dumps(args)})")

            if name == "inspect_metadata":
                result = inspect_metadata(df)
            elif name == "get_column_stats":
                result = get_column_stats(df, args["col"])
            elif name == "create_interaction":
                df, m = create_interaction(df, args["expression"], args["new_col_name"])
                result = {"result": m}; log(f"  → {m}")
            elif name == "encode_categorical":
                df, m = encode_categorical(df, args["col"], args.get("method", "onehot"))
                result = {"result": m}; log(f"  → {m}")
            elif name == "correlation_analysis":
                result = correlation_analysis(df, args["target"])
            elif name == "select_top_features":
                df, m = select_top_features(df, args["target"], args["k"])
                result = {"result": m}; log(f"  → {m}")
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    eng_path = os.path.join(output_dir, "engineered_data.csv")
    df.to_csv(eng_path, index=False)
    log(f"\n[Agent 2]: Saved → {eng_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    log(f"[Agent 2]: Columns: {list(df.columns)}")
    return eng_path, summary