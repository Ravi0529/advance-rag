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


def generate_subqueries(user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return json.loads(response.choices[0].message.content)


def parallel_query(
    user_query, collection_name="parallel_query", qdrant_url="http://localhost:6333"
):
    sub_queries = generate_subqueries(user_query)
    # print(sub_queries)

    retriever = get_retriever(qdrant_url=qdrant_url, collection_name=collection_name)

    retrieved_chunks = set()
    for sub_query in sub_queries:
        results = retriever.similarity_search(sub_query)
        for chunk in results:
            retrieved_chunks.add(chunk.page_content)

    return list(retrieved_chunks)
