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


def generate_hypothetical_doc(user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Generate a hypothetical answer or passage that would ideally answer this question: {user_query}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return json.loads(response.choices[0].message.content)


def hyde(
    user_query,
    collection_name="hyde",
    qdrant_url="http://localhost:6333",
):
    hypothetical_answer = generate_hypothetical_doc(user_query)
    retriever = get_retriever(qdrant_url, collection_name)

    results = retriever.similarity_search(hypothetical_answer["answer"])
    return [doc.page_content for doc in results]


def get_final_answer(user_query, chunks):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that uses the retrieved documents to answer the user's query based on a hypothetical ideal answer.",
        },
        {
            "role": "user",
            "content": f"User asked: {user_query}\n\nHere are related passages based on a hypothetical answer:\n\n{chr(10).join(chunks)}",
        },
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content
