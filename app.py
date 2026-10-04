import os
import sys
import json
from typing import Any, Dict, List, Union
from dotenv import load_dotenv
from groq import Groq, GroqError
from flask import Flask, request, jsonify, render_template_string

# Ensure UTF-8 output encoding for Windows terminal / PowerShell compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Initialize Flask application top-level export for Vercel Serverless Function deployment
app = Flask(__name__)


def get_sanitized_api_key() -> str:
    """Loads and sanitizes GROQ_API_KEY from environment, stripping whitespace/quotes."""
    load_dotenv(override=True)
    raw_key = os.getenv("GROQ_API_KEY", "")
    if not raw_key:
        return ""
    clean_key = raw_key.strip().strip('"\'')
    return clean_key


def mask_api_key(key: str) -> str:
    """Returns a safe masked version of the API key for logging without revealing secret value."""
    if not key:
        return "<MISSING>"
    if len(key) <= 8:
        return f"{key[:2]}*** (length: {len(key)})"
    return f"{key[:4]}...{key[-4:]} (length: {len(key)})"


# Configuration
DEFAULT_MODEL = "qwen/qwen3.8-27b"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are a helpful AI Calculator Assistant. "
    "When a user asks you to perform a mathematical operation (addition, subtraction, multiplication, division), "
    "you MUST use the provided 'calculate' tool to get the accurate result. "
    "Do NOT perform manual arithmetic calculation in your head. "
    "Always rely on the tool output for math results and present the final answer clearly to the user."
)


# =====================================================================
# 1. TOOL FUNCTION DEFINITION
# =====================================================================
def calculate(operation: str, a: Union[int, float], b: Union[int, float]) -> Dict[str, Any]:
    """Executes basic arithmetic operations safely without using eval() or exec()."""
    if not isinstance(operation, str):
        return {"error": "Invalid or missing 'operation' string parameter."}

    if isinstance(a, bool) or isinstance(b, bool):
        return {"error": "Invalid numerical arguments: Booleans are not allowed."}

    try:
        num_a = float(a)
        num_b = float(b)
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid numerical arguments: {e}"}

    op = operation.strip().lower()

    if op == "add":
        res = num_a + num_b
    elif op == "subtract":
        res = num_a - num_b
    elif op == "multiply":
        res = num_a * num_b
    elif op == "divide":
        if num_b == 0:
            return {"error": "Division by zero is not allowed."}
        res = num_a / num_b
    else:
        return {
            "error": f"Invalid operation '{operation}'. Supported operations are: 'add', 'subtract', 'multiply', 'divide'."
        }

    if res.is_integer():
        res = int(res)

    return {"result": res}


ALLOWED_TOOLS = {
    "calculate": calculate
}


# =====================================================================
# 2. GROQ TOOL DEFINITION SCHEMA
# =====================================================================
TOOLS: Any = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs basic arithmetic operations (addition, subtraction, multiplication, division) on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform."
                    },
                    "a": {
                        "type": "number",
                        "description": "The first numeric operand."
                    },
                    "b": {
                        "type": "number",
                        "description": "The second numeric operand."
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
]


# =====================================================================
# 3. CORE AGENT LOOP & AGENT EXECUTOR
# =====================================================================
def execute_agent_loop(client: Groq, model: str, user_prompt: str) -> Dict[str, Any]:
    """
    Executes the single-agent tool call loop and returns structured output for Web & CLI.
    """
    messages: List[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    iteration = 0
    raw_tool_calls_log = []
    tool_results_log = []

    while iteration < MAX_ITERATIONS:
        iteration += 1

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,  # type: ignore
                tool_choice="auto",
                temperature=0.0
            )
        except GroqError as e:
            return {"error": f"Groq API Error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected Error: {e}"}

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls is not None and len(tool_calls) > 0:
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            }
            messages.append(assistant_msg)

            for tool_call in tool_calls:
                tool_call_id = tool_call.id
                func_name = tool_call.function.name
                raw_args_str = tool_call.function.arguments

                raw_tool_calls_log.append({
                    "id": tool_call_id,
                    "name": func_name,
                    "arguments": raw_args_str
                })

                if func_name not in ALLOWED_TOOLS:
                    result: Dict[str, Any] = {"error": f"Tool '{func_name}' is not allowed."}
                else:
                    try:
                        args = json.loads(raw_args_str)
                    except json.JSONDecodeError as err:
                        result = {"error": f"Invalid JSON arguments: {err}"}
                        args = None

                    if args is not None:
                        if not isinstance(args, dict):
                            result = {"error": "Tool arguments must be a JSON object."}
                        else:
                            func = ALLOWED_TOOLS[func_name]
                            result = func(**args)  # type: ignore[arg-type]

                tool_results_log.append({
                    "id": tool_call_id,
                    "result": result
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result)
                })

            continue

        final_answer = response_message.content or "No response generated."
        return {
            "success": True,
            "raw_tool_calls": raw_tool_calls_log,
            "tool_results": tool_results_log,
            "final_answer": final_answer
        }

    return {
        "success": False,
        "error": f"Reached maximum iteration limit ({MAX_ITERATIONS}) without completing task."
    }


def run_agent_loop(client: Groq, model: str, user_prompt: str) -> None:
    """CLI Agent Loop printing raw logs directly to console."""
    output = execute_agent_loop(client, model, user_prompt)
    if "error" in output and not output.get("success"):
        print(f"\n❌ Error: {output['error']}\n")
        return

    for tc in output.get("raw_tool_calls", []):
        print("\n==================== RAW TOOL CALL ====================")
        print(f"Tool Call ID : {tc['id']}")
        print(f"Tool Name    : {tc['name']}")
        print(f"Raw Arguments: {tc['arguments']}")
        print("=======================================================")
        
        try:
            parsed_args = json.loads(tc['arguments'])
            args_formatted = ", ".join(f"{k}={repr(v)}" for k, v in parsed_args.items())
            print(f"⚙️ Executing Python function: {tc['name']}({args_formatted})\n")
        except Exception:
            print("")

    for tr in output.get("tool_results", []):
        print(f"📊 Tool Output Result: {json.dumps(tr['result'])}")

    print(f"\n🤖 Agent Response:\n{output.get('final_answer', '')}\n")


# =====================================================================
# 4. FLASK WEB ROUTES & WEB INTERFACE (For Vercel Deployment)
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Calculator Agent - SkillAudit.ai</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-emerald: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --code-bg: #090d16;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top, #1e1b4b 0%, #0f172a 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 850px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            margin-bottom: 2rem;
        }

        .form-group {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        input[type="text"] {
            flex: 1;
            padding: 0.85rem 1.2rem;
            border-radius: 10px;
            border: 1px solid var(--card-border);
            background: rgba(15, 23, 42, 0.6);
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: all 0.2s;
        }

        input[type="text"]:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        button {
            padding: 0.85rem 1.5rem;
            border-radius: 10px;
            border: none;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.15s, opacity 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        button:hover { opacity: 0.9; transform: translateY(-1px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .presets {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.825rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .preset-btn:hover {
            background: rgba(139, 92, 246, 0.2);
            color: var(--text-main);
            border-color: var(--accent-purple);
        }

        .output-box {
            display: none;
        }

        .section-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.6rem;
            font-weight: 600;
        }

        .code-block {
            background: var(--code-bg);
            border-radius: 10px;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.875rem;
            color: #38bdf8;
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1.25rem;
            white-space: pre-wrap;
        }

        .agent-answer {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 10px;
            padding: 1.25rem;
            color: #ecfdf5;
            font-size: 1.1rem;
            font-weight: 500;
        }

        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Calculator Agent</h1>
            <p>SkillAudit.ai Week 1 - Function & Tool Calling Agent powered by Groq</p>
        </div>

        <div class="card">
            <div class="presets">
                <span style="font-size: 0.8rem; color: var(--text-muted); align-self: center; margin-right: 0.25rem;">Try:</span>
                <button type="button" class="preset-btn" onclick="setPrompt('What is 15 + 27?')">15 + 27</button>
                <button type="button" class="preset-btn" onclick="setPrompt('Subtract 45 from 100')">100 - 45</button>
                <button type="button" class="preset-btn" onclick="setPrompt('Multiply 25 by 16')">25 × 16</button>
                <button type="button" class="preset-btn" onclick="setPrompt('Divide 144 by 12')">144 ÷ 12</button>
                <button type="button" class="preset-btn" onclick="setPrompt('Divide 50 by 0')">Divide by 0</button>
                <button type="button" class="preset-btn" onclick="setPrompt('What is the capital of France?')">Capital of France</button>
            </div>

            <form id="agentForm">
                <div class="form-group">
                    <input type="text" id="promptInput" placeholder="Ask a calculation or question..." required>
                    <button type="submit" id="submitBtn">
                        <span id="btnText">Calculate 🚀</span>
                        <span id="btnSpinner" class="spinner" style="display: none;"></span>
                    </button>
                </div>
            </form>
        </div>

        <div id="outputContainer" class="card output-box">
            <div id="toolSection" style="display:none;">
                <div class="section-title">⚙️ Raw Tool Call (Function Call)</div>
                <div id="toolCallCode" class="code-block"></div>

                <div class="section-title">📊 Tool Execution Result</div>
                <div id="toolResultCode" class="code-block" style="color: #a78bfa;"></div>
            </div>

            <div class="section-title">🤖 Agent Response</div>
            <div id="agentAnswer" class="agent-answer"></div>
        </div>
    </div>

    <script>
        function setPrompt(text) {
            document.getElementById('promptInput').value = text;
        }

        document.getElementById('agentForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const input = document.getElementById('promptInput');
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const outputContainer = document.getElementById('outputContainer');
            const toolSection = document.getElementById('toolSection');
            const toolCallCode = document.getElementById('toolCallCode');
            const toolResultCode = document.getElementById('toolResultCode');
            const agentAnswer = document.getElementById('agentAnswer');

            const prompt = input.value.trim();
            if (!prompt) return;

            submitBtn.disabled = true;
            btnText.innerText = "Thinking...";
            btnSpinner.style.display = "inline-block";
            outputContainer.style.display = "none";

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });

                const data = await res.json();
                outputContainer.style.display = "block";

                if (data.error) {
                    toolSection.style.display = "none";
                    agentAnswer.innerText = "❌ Error: " + data.error;
                    agentAnswer.style.borderColor = "rgba(239, 68, 68, 0.4)";
                    agentAnswer.style.background = "rgba(239, 68, 68, 0.1)";
                } else {
                    agentAnswer.style.borderColor = "rgba(16, 185, 129, 0.3)";
                    agentAnswer.style.background = "rgba(16, 185, 129, 0.1)";
                    agentAnswer.innerText = data.final_answer;

                    if (data.raw_tool_calls && data.raw_tool_calls.length > 0) {
                        toolSection.style.display = "block";
                        toolCallCode.innerText = JSON.stringify(data.raw_tool_calls, null, 2);
                        toolResultCode.innerText = JSON.stringify(data.tool_results, null, 2);
                    } else {
                        toolSection.style.display = "none";
                    }
                }
            } catch (err) {
                outputContainer.style.display = "block";
                toolSection.style.display = "none";
                agentAnswer.innerText = "❌ Failed to communicate with server: " + err.message;
            } finally {
                submitBtn.disabled = false;
                btnText.innerText = "Calculate 🚀";
                btnSpinner.style.display = "none";
            }
        });
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    """Serves the interactive web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """API endpoint for Vercel web client."""
    data = request.get_json(silent=True) or {}
    user_prompt = data.get("prompt", "").strip()

    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    api_key = get_sanitized_api_key()
    if not api_key or api_key == "your_groq_api_key_here":
        return jsonify({"error": "Valid GROQ_API_KEY environment variable is missing on Vercel server."}), 500

    raw_env_model = os.getenv("GROQ_MODEL")
    model = (raw_env_model if raw_env_model else DEFAULT_MODEL).strip().strip('"\'')

    try:
        client = Groq(api_key=api_key)
        result = execute_agent_loop(client, model, user_prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================================
# 5. CLI INTERACTIVE INTERFACE
# =====================================================================
def main() -> None:
    api_key = get_sanitized_api_key()
    if not api_key or api_key == "your_groq_api_key_here":
        print("\n❌ Error: Valid GROQ_API_KEY is missing!")
        print("Please set your API key in your '.env' file.")
        print("Example .env file content:\n  GROQ_API_KEY=gsk_your_actual_api_key_here\n")
        sys.exit(1)

    raw_env_model = os.getenv("GROQ_MODEL")
    model = (raw_env_model if raw_env_model else DEFAULT_MODEL).strip().strip('"\'')

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"\n❌ Error initializing Groq client: {e}")
        sys.exit(1)

    print("==================================================================")
    print("🤖 Welcome to SkillAudit.ai Week 1 AI Calculator Agent CLI")
    print(f"📌 Model in use: {model}")
    print(f"📌 Key status  : Loaded ({mask_api_key(api_key)})")
    print("📌 Type your prompt or mathematical question below.")
    print("📌 Type 'exit' or 'quit' to close the program.")
    print("==================================================================\n")

    while True:
        try:
            user_input = input("User > ").strip()

            if len(user_input) > 2000:
                print("\n❌ Input is too long. Please keep your prompt under 2000 characters.\n")
                continue

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break

            run_agent_loop(client, model, user_input)

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Program interrupted. Goodbye!")
            break


if __name__ == "__main__":
    # If run locally via CLI: start CLI mode
    main()
