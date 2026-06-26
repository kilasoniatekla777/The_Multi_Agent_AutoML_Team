
import json, os, sys
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.agent_tools import execute_python_code

MODEL = "gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute a complete Python script and return stdout/stderr. "
                "Your code MUST print metrics on a line in EXACTLY this format:\n"
                "METRICS: accuracy=0.82 f1=0.81 recall=0.80\n"
                "Import everything inside the code string. Use warnings.filterwarnings('ignore')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code_string": {
                        "type": "string",
                        "description": "Complete self-contained Python code to run",
                    }
                },
                "required": ["code_string"],
            },
        },
    },
]


def run_agent3(engineered_csv_path, agent1_summary, agent2_summary, target_column, output_dir, log_lines):
    client = OpenAI()

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("\n" + "="*60)
    log("AGENT 3 — THE MODEL TRAINER  ('The Coder')")
    log("="*60)

    system_prompt = f"""You are an expert ML Engineer Agent.
Train an XGBoost classifier on the pre-processed dataset.
- Data file: '{engineered_csv_path}'
- Target column: '{target_column}'

Workflow:
1. Write complete Python code to:
   - Load the CSV
   - Split 80/20 train/test with random_state=42
   - Train XGBoostClassifier
   - Print metrics in EXACTLY this format (one line, nothing else on that line):
     METRICS: accuracy=X.XX f1=X.XX recall=X.XX
   Use: from sklearn.metrics import accuracy_score, f1_score, recall_score
   Use weighted average for f1 and recall.

2. Call execute_python_code.

3. Read the output and decide:
   - If accuracy >= 0.78, OR you have tried 3+ configurations: STOP and write your final report.
   - If performance is poor: explain WHY and write NEW code with different hyperparameters.

Hyperparameters to tune if needed:
  n_estimators: 100/200/300, max_depth: 3/5/7,
  learning_rate: 0.1/0.05/0.01, subsample: 0.8/1.0,
  scale_pos_weight for class imbalance.

Always import everything inside the code string.

When finished, write your report in EXACTLY this format:
TRAINING REPORT:
- Best accuracy: X.XX
- Best F1: X.XX
- Best recall: X.XX
- Winning hyperparameters: <list>
- Iterations tried: <N>
- Decision rationale: <why you stopped>"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"Agent 1 summary:\n{agent1_summary}\n\n"
            f"Agent 2 summary:\n{agent2_summary}\n\n"
            f"Train an XGBoost model on '{engineered_csv_path}' predicting '{target_column}'. "
            f"Start with a baseline, then iterate if needed."
        )},
    ]

    best_metrics = {}
    best_code = ""
    iteration = 0

    for _ in range(20):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            log(f"\n[Agent 3 thinks]: {msg.content}")
            if "TRAINING REPORT:" in msg.content:
                best_metrics["report"] = msg.content

        if not msg.tool_calls:
            log("\n[Agent 3]: Finished training.")
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "execute_python_code":
                iteration += 1
                code = args["code_string"]
                log(f"\n[Agent 3 executes code — Iteration {iteration}]")
                log("─" * 40)
                for line in code.split("\n"):
                    if any(k in line for k in ["XGBClassifier", "n_estimators", "max_depth",
                                                "learning_rate", "subsample", "METRICS"]):
                        log(f"  {line.strip()}")
                log("─" * 40)

                result = execute_python_code(code)

                if result["success"]:
                    log(f"[Agent 3 stdout]: {result['stdout'].strip()}")
                    for line in result["stdout"].split("\n"):
                        if line.startswith("METRICS:"):
                            for part in line.replace("METRICS:", "").strip().split():
                                k, v = part.split("=")
                                best_metrics[k] = float(v)
                            best_code = code
                            log(f"[Agent 3 metrics]: {best_metrics}")
                else:
                    log(f"[Agent 3 ERROR]: {result['error']}")
                    result["stdout"] = f"EXECUTION ERROR:\n{result['error']}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": f"Unknown tool: {name}"}),
                })

    if best_code:
        code_path = os.path.join(output_dir, "best_model_code.py")
        with open(code_path, "w") as f:
            f.write(best_code)
        log(f"\n[Agent 3]: Saved best model code → {code_path}")

    best_metrics["iterations"] = iteration
    return best_metrics