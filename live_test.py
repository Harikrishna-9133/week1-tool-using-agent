import sys
from app import get_sanitized_api_key, run_agent_loop, DEFAULT_MODEL
from groq import Groq

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    api_key = get_sanitized_api_key()
    client = Groq(api_key=api_key)
    model = DEFAULT_MODEL

    test_prompts = [
        "What is 15 + 27?",
        "Subtract 45 from 100",
        "What is 25 multiplied by 16?",
        "Divide 144 by 12",
        "Divide 50 by 0",
        "What is the capital of France?"
    ]

    print("=====================================================")
    print("🚀 Running Live Groq Tool Calling Tests...")
    print("=====================================================\n")

    for idx, prompt in enumerate(test_prompts, 1):
        print(f"--- TEST CASE {idx}: '{prompt}' ---")
        run_agent_loop(client, model, prompt)
        print("\n")


if __name__ == "__main__":
    main()

