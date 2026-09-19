import os
import sys
import json
from typing import Any, Dict, List, Union
from dotenv import load_dotenv
from groq import Groq, GroqError

# Ensure UTF-8 output encoding for Windows terminal / PowerShell compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_sanitized_api_key() -> str:
    """Loads and sanitizes GROQ_API_KEY from environment, stripping whitespace/quotes."""
    # Force reload of .env with override=True to capture changes immediately
    load_dotenv(override=True)
    raw_key = os.getenv("GROQ_API_KEY", "")
    if not raw_key:
        return ""
    # Strip whitespace, newlines, and potential quotes
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


# System prompt guiding the agent's behavior
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
    """
    Executes basic arithmetic operations safely without using eval() or exec().

    Args:
        operation (str): Operation name ('add', 'subtract', 'multiply', 'divide')
        a (Union[int, float]): First numeric operand
        b (Union[int, float]): Second numeric operand

    Returns:
        Dict[str, Any]: Structured dictionary containing 'result' or 'error'
    """
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

    # Format cleanly if result is a whole number integer
    if res.is_integer():
        res = int(res)

    return {"result": res}


# Mapping of allowed python tool functions
ALLOWED_TOOLS = {
    "calculate": calculate
}

# =====================================================================
# 2. GROQ TOOL DEFINITION SCHEMA (OpenAI-compatible)
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
# 3. CORE AGENT LOOP
# =====================================================================
def run_agent_loop(client: Groq, model: str, user_prompt: str) -> None:
    """
    Executes the single-agent tool call loop:
    User input -> Groq LLM -> Raw Tool Call -> Tool Execution -> Tool Result back to Groq -> Final Answer
    """
    messages: List[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        try:
            # Call Groq API with tool definitions and auto choice
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,  # type: ignore
                tool_choice="auto",
                temperature=0.0  # Deterministic output for tool calls
            )

        except GroqError as e:
            if "401" in str(e) or "invalid_api_key" in str(e):
                print("\n❌ [API Error 401] Invalid Groq API Key.")
                print("💡 Please verify that your API key in '.env' is active and copied correctly from https://console.groq.com/keys\n")
            elif "404" in str(e) or "model_not_found" in str(e):
                print(f"\n❌ [API Error 404] Model '{model}' not found or not accessible on Groq API.")
                print("💡 Please check GROQ_MODEL in your '.env' file.\n")
            else:
                print(f"\n❌ [API Error] Groq API call failed: {e}\n")
            return
        except Exception as e:
            print(f"\n❌ [Unexpected Error] {e}\n")
            return

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Case A: Model decides to call one or more tools
        if tool_calls is not None and len(tool_calls) > 0:
            # Append normalized assistant message with tool calls to conversation history
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

                # STRICT REQ #6: Print raw tool call before executing it
                print("\n==================== RAW TOOL CALL ====================")
                print(f"Tool Call ID : {tool_call_id}")
                print(f"Tool Name    : {func_name}")
                print(f"Raw Arguments: {raw_args_str}")
                print("=======================================================\n")

                # STRICT REQ #7: Execute only allowed functions
                if func_name not in ALLOWED_TOOLS:
                    result: Dict[str, Any] = {"error": f"Tool '{func_name}' is not allowed or supported."}
                else:
                    # Safely parse JSON arguments
                    try:
                        args = json.loads(raw_args_str)
                    except json.JSONDecodeError as err:
                        result = {"error": f"Invalid JSON arguments provided by model: {err}"}
                        args = None

                    if args is not None:
                        operation = args.get("operation")
                        a = args.get("a")
                        b = args.get("b")
                        print(f"⚙️ Executing Python function: calculate(operation='{operation}', a={a}, b={b})")
                        result = ALLOWED_TOOLS[func_name](operation=operation, a=a, b=b)

                print(f"📊 Tool Output Result: {json.dumps(result)}")

                # STRICT REQ #8: Send assistant tool message and tool result back using tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result)
                })

            # Continue loop to send tool output back to model
            continue

        # Case B: Model returns final natural language response
        final_answer = response_message.content
        print(f"\n🤖 Agent Response:\n{final_answer}\n")
        return

    print(f"\n⚠️ Reached maximum iteration limit ({MAX_ITERATIONS}) without completing the task.\n")


# =====================================================================
# 4. CLI INTERACTIVE INTERFACE
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
    main()



