import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

prompt_path = Path(__file__).parent / "data" / "system_prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as file:
    system_prompt = file.read()


def step_back_reformulation(user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Given this specific question, reformulate it into a more general or broader question that will help retrieve useful background knowledge: {user_query}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return json.loads(response.choices[0].message.content)


def step_back_prompting(user_query):
    general_query = step_back_reformulation(user_query)
    # print(general_query)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": general_query},
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return json.loads(response.choices[0].message.content)


def get_final_answer(user_query, chunks):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers user queries using both background knowledge and direct context.",
        },
        {
            "role": "user",
            "content": f"User asked: {user_query}\n\nHere is some useful background knowledge:\n\n{chr(10).join(chunks)}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content
