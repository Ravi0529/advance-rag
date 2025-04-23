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


def reciprocal_rank_fusion(
    user_query,
    collection_name="reciprocal_rank_fusion",
    qdrant_url="http://localhost:6333",
    k=60,
):
    sub_queries = generate_subqueries(user_query)
    retriever = get_retriever(qdrant_url, collection_name)

    scored_docs = {}
    for sub_query in sub_queries:
        results = retriever.similarity_search(sub_query, k=k)
        for rank, doc in enumerate(results, start=1):
            content = doc.page_content
            score = 1 / (rank + 60)
            scored_docs[content] = scored_docs.get(content, 0) + score

    sorted_docs = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in sorted_docs]


def get_final_answer(user_query, chunks):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that uses provided context to answer queries.",
        },
        {
            "role": "user",
            "content": f"User asked: {user_query}\n\nHere is some information:\n\n{'\n\n'.join(chunks)}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content
