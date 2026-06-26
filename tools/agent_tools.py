"""
agent_tools.py
All tool functions available to the three agents.
Each function takes a DataFrame (and optional args), performs an operation,
and returns a result that the LLM can read and reason about.
"""

import pandas as pd
import numpy as np
import io
import sys
import traceback


# ─────────────────────────────────────────────
#  AGENT 1 TOOLS  (Data Cleaner)
# ─────────────────────────────────────────────

def inspect_metadata(df: pd.DataFrame) -> dict:
    """
    Returns shape, dtypes, and null counts.
    The LLM reads this to decide what needs fixing.
    """
    null_counts = df.isnull().sum().to_dict()
    null_pct    = (df.isnull().mean() * 100).round(2).to_dict()
    return {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": null_counts,
        "null_pct": null_pct,
    }


def get_column_stats(df: pd.DataFrame, col: str) -> dict:
    """
    For numeric cols: min/max/mean/std/unique count.
    For object cols: top values and cardinality.
    """
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}

    if pd.api.types.is_numeric_dtype(df[col]):
        return {
            "type": "numeric",
            "min":    float(df[col].min())  if not df[col].isnull().all() else None,
            "max":    float(df[col].max())  if not df[col].isnull().all() else None,
            "mean":   float(df[col].mean()) if not df[col].isnull().all() else None,
            "std":    float(df[col].std())  if not df[col].isnull().all() else None,
            "unique": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
        }
    else:
        top = df[col].value_counts().head(10).to_dict()
        return {
            "type": "categorical",
            "unique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in top.items()},
            "null_count": int(df[col].isnull().sum()),
        }


def impute_missing(df: pd.DataFrame, col: str, strategy: str) -> tuple[pd.DataFrame, str]:
    """
    Fills NaN values in `col`.
    strategy: 'mean' | 'median' | 'mode' | 'drop_rows'
    Returns the updated DataFrame and a log message.
    """
    if col not in df.columns:
        return df, f"ERROR: Column '{col}' not found"

    before = int(df[col].isnull().sum())
    if before == 0:
        return df, f"No nulls in '{col}', skipped"

    df = df.copy()
    if strategy == "mean":
        val = df[col].mean()
        df[col].fillna(val, inplace=True)
        msg = f"Imputed '{col}' with mean={val:.4f} ({before} nulls filled)"
    elif strategy == "median":
        val = df[col].median()
        df[col].fillna(val, inplace=True)
        msg = f"Imputed '{col}' with median={val:.4f} ({before} nulls filled)"
    elif strategy == "mode":
        val = df[col].mode()[0]
        df[col].fillna(val, inplace=True)
        msg = f"Imputed '{col}' with mode='{val}' ({before} nulls filled)"
    elif strategy == "drop_rows":
        df.dropna(subset=[col], inplace=True)
        msg = f"Dropped rows with null '{col}' ({before} rows removed)"
    else:
        msg = f"Unknown strategy '{strategy}' — no change"

    return df, msg


def drop_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, str]:
    """Drops a column. Returns updated df and log message."""
    if col not in df.columns:
        return df, f"ERROR: Column '{col}' not found"
    df = df.drop(columns=[col])
    return df, f"Dropped column '{col}'"


# ─────────────────────────────────────────────
#  AGENT 2 TOOLS  (Feature Engineer)
# ─────────────────────────────────────────────

def create_interaction(df: pd.DataFrame, expression: str, new_col_name: str) -> tuple[pd.DataFrame, str]:
    """
    Creates a new column by evaluating `expression` using df.eval().
    e.g. expression="Fare / (Age + 1)", new_col_name="fare_per_age"
    """
    try:
        df = df.copy()
        df[new_col_name] = df.eval(expression)
        return df, f"Created feature '{new_col_name}' = {expression}"
    except Exception as e:
        return df, f"ERROR creating feature: {e}"


def encode_categorical(df: pd.DataFrame, col: str, method: str = "onehot") -> tuple[pd.DataFrame, str]:
    """
    Encodes a categorical column.
    method: 'onehot' | 'label'
    """
    if col not in df.columns:
        return df, f"ERROR: Column '{col}' not found"
    df = df.copy()
    if method == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df, f"One-hot encoded '{col}' → {list(dummies.columns)}"
    elif method == "label":
        df[col] = df[col].astype("category").cat.codes
        return df, f"Label-encoded '{col}'"
    else:
        return df, f"Unknown encoding method '{method}'"


def correlation_analysis(df: pd.DataFrame, target: str) -> dict:
    """
    Returns Pearson correlation of all numeric features with the target column.
    """
    if target not in df.columns:
        return {"error": f"Target '{target}' not found"}
    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        return {"error": f"Target '{target}' is not numeric"}
    corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    return {"correlations_with_target": corr.round(4).to_dict()}


def select_top_features(df: pd.DataFrame, target: str, k: int) -> tuple[pd.DataFrame, str]:
    """
    Keeps the k numeric features most correlated with target, plus target itself.
    """
    if target not in df.columns:
        return df, f"ERROR: Target '{target}' not found"
    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        return df, f"ERROR: Target '{target}' is not numeric after current transforms"
    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    top_cols = list(corr.head(k).index) + [target]
    df = df[top_cols]
    return df, f"Selected top {k} features: {list(corr.head(k).index)}"


# ─────────────────────────────────────────────
#  AGENT 3 TOOLS  (Model Trainer)
# ─────────────────────────────────────────────

def execute_python_code(code_string: str) -> dict:
    """
    Executes a Python code string in a sandboxed scope.
    Returns stdout, stderr, and any exception info.
    The generated code should print its metrics to stdout.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    exec_globals = {}

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        exec(code_string, exec_globals)
        success = True
        error = None
    except Exception:
        success = False
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return {
        "success": success,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "error":  error,
    }