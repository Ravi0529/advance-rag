from retriever import get_retriever


def ask_nodejs_doc(query: str) -> str:
    retriever = get_retriever("http://localhost:6333", "basic_rag")
    results = retriever.similarity_search(query=query)

    if not results:
        return (
            "Sorry, I couldn't find anything related to your query in the Node.js docs."
        )

    response = "Here's what I found in the Node.js documentation:\n\n"
    for i, result in enumerate(results, 1):
        response += f"{i}. {result.page_content.strip()}\n\n"

    return response.strip()
