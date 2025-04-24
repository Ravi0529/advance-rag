import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from retriever import get_retriever
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

prompt_path = Path(__file__).parent / "data" / "system_prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as file:
    system_prompt = file.read()


def decompose_query(user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Break down this question into logical steps or sub-questions: {user_query}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return json.loads(response.choices[0].message.content)


def query_decomposition(
    user_query,
    collection_name="query_decomposition",
    qdrant_url="http://localhost:6333",
    max_steps=5,
):
    sub_steps = decompose_query(user_query)
    retriever = get_retriever(qdrant_url, collection_name)

    collected_chunks = set()
    for step in sub_steps[:max_steps]:
        results = retriever.similarity_search(step)
        for doc in results:
            collected_chunks.add(doc.page_content)

    return list(collected_chunks)


def get_final_answer(user_query, chunks):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that uses provided context to answer queries step-by-step using reasoning.",
        },
        {
            "role": "user",
            "content": f"User asked: {user_query}\n\nHere is some information from each logical step:\n\n{chr(10).join(chunks)}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content
