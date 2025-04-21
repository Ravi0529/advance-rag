import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from tools.nodejs_qa import ask_nodejs_doc

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

available_tools = {
    "ask_nodejs_doc": {
        "fn": ask_nodejs_doc,
        "description": "Takes a question about Node.js and returns relevant documentation info",
    },
}

system_prompt = f"""
You are a helpful AI Assistant specialized in resolving queries using available tools.

You work step-by-step: plan → action → observe → output.

Available Tools:
- ask_nodejs_doc: Takes a question about Node.js and returns related answers from docs.

Output JSON Format:
{{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
}}

Example:
User Query: What is the FS module?
Output: {{ "step": "plan", "content": "The user is asking about Node.js FS module" }}
Output: {{ "step": "plan", "content": "I should use ask_nodejs_doc" }}
Output: {{ "step": "action", "function": "ask_nodejs_doc", "input": "What is FS module?" }}
Output: {{ "step": "observe", "output": "..." }}
Output: {{ "step": "output", "content": "The FS module allows working with the file system..." }}
"""

messages = [{"role": "system", "content": system_prompt}]

while True:
    user_query = input("> ")
    if user_query.strip().lower() == "clear":
        break
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            response_format={"type": "json_object"},
            n=1,
            messages=messages,
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_output)})

        if parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get('content')}")
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if tool := available_tools.get(tool_name):
                result = tool["fn"](tool_input)
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "output": result}),
                    }
                )
                continue

        if parsed_output.get("step") == "output":
            print(f"\n🤖 Final Answer:\n{parsed_output.get('content')}\n")
            break
