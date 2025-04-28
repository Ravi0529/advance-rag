import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def semantic_routing(user_query: str) -> str | None:
    system_prompt = (
        "You are a domain classifier for technical queries.\n"
        "Your task is to strictly classify a given user query into one of the following categories:\n"
        "- If the query is primarily about Node.js (server-side JavaScript), reply exactly: 'nodejs'\n"
        "- If the query is primarily about Python (the programming language), reply exactly: 'python'\n"
        "- If the query is unclear, ambiguous, or about another topic, reply exactly: 'unknown'\n\n"
        "Important:\n"
        "- Reply with only one word: 'nodejs', 'python', or 'unknown'.\n"
        "- Do not provide any explanation, reasoning, or extra text.\n"
        "- Be strict: if unsure, prefer 'unknown'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "text"},
        n=1,
        messages=messages,
    )

    category = response.choices[0].message.content.strip().lower()

    if category == "nodejs":
        return "nodejs-collection"
    elif category == "python":
        return "python-collection"
    else:
        return None
