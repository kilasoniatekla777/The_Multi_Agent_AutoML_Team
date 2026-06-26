
import json, os, sys
import pandas as pd
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.agent_tools import inspect_metadata, get_column_stats, impute_missing, drop_column

MODEL = "gpt-4o-mini"   

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_metadata",
            "description": (
                "Inspect the dataset: returns shape, column names, data types, "
                "null counts and null percentages. Call this FIRST."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_stats",
            "description": "Get detailed stats for one column (min/max/mean for numeric, top values for categorical).",
            "parameters": {
                "type": "object",
                "properties": {
                    "col": {"type": "string", "description": "Column name to inspect"},
                },
                "required": ["col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "impute_missing",
            "description": (
                "Fill NaN values in a column. "
                "Use 'mean' or 'median' for numeric, 'mode' for categorical, 'drop_rows' to remove rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col":      {"type": "string"},
                    "strategy": {"type": "string", "enum": ["mean", "median", "mode", "drop_rows"]},
                },
                "required": ["col", "strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drop_column",
            "description": (
                "Remove a column that is unusable: unique IDs, free-text names, "
                "ticket codes, or columns with >60% missing values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col": {"type": "string", "description": "Column name to drop"},
                },
                "required": ["col"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a meticulous Data Cleaning Agent.
Your job is to inspect a dataset, identify quality issues, and fix them using your tools.

Rules:
- Always start by calling inspect_metadata.
- Use get_column_stats on any suspicious column before acting.
- Drop columns that are unique identifiers (cardinality == row count), free-text names, or ticket codes.
- For numeric columns with <30% missing: impute with median.
- For categorical columns with <30% missing: impute with mode.
- For columns with >60% missing: drop them.
- Be explicit about WHY you make each decision.
- When satisfied, stop calling tools and write a final summary in EXACTLY this format:

CLEANING SUMMARY:
- <action> | Reason: <why>
- ...
COLUMNS REMAINING: <comma-separated list>"""


def run_agent1(raw_csv_path: str, output_dir: str, log_lines: list) -> tuple[str, str]:
    client = OpenAI()  
    df = pd.read_csv(raw_csv_path)

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("\n" + "="*60)
    log("AGENT 1 — THE DATA CLEANER  ('The Auditor')")
    log("="*60)
    log(f"Loaded: {raw_csv_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Please clean this dataset. It has {df.shape[0]} rows and "
            f"{df.shape[1]} columns. Use your tools to inspect and fix it."
        )},
    ]

    summary = ""

    for _ in range(20):  # max turns
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)  # append assistant message

       
        if msg.content:
            log(f"\n[Agent 1 thinks]: {msg.content}")
            if "CLEANING SUMMARY:" in msg.content:
                summary = msg.content

 
        if not msg.tool_calls:
            log("\n[Agent 1]: Finished cleaning.")
            break

    
        for tc in msg.tool_calls:
            name  = tc.function.name
            args  = json.loads(tc.function.arguments)
            log(f"\n[Agent 1 calls tool]: {name}({json.dumps(args)})")

            if name == "inspect_metadata":
                result = inspect_metadata(df)
            elif name == "get_column_stats":
                result = get_column_stats(df, args["col"])
            elif name == "impute_missing":
                df, msg_txt = impute_missing(df, args["col"], args["strategy"])
                result = {"result": msg_txt}
                log(f"  → {msg_txt}")
            elif name == "drop_column":
                df, msg_txt = drop_column(df, args["col"])
                result = {"result": msg_txt}
                log(f"  → {msg_txt}")
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result),
            })

   
    clean_path = os.path.join(output_dir, "clean_data.csv")
    df.to_csv(clean_path, index=False)
    log(f"\n[Agent 1]: Saved → {clean_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return clean_path, summary