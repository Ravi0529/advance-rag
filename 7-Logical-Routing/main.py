from HyDE import hyde, get_final_answer


def logical_routing(user_query):
    user_query_lower = user_query.lower()

    if "python" in user_query_lower:
        collection = "python-collection"
    elif "nodejs" in user_query_lower or "node.js" in user_query_lower:
        collection = "nodejs-collection"
    else:
        collection = None

    return collection


def main():
    while True:
        query = input("> ")
        if query.lower() == "clear":
            break

        collection_name = logical_routing(query)

        if collection_name is None:
            print("Please specify whether your query is about Node.js or Python.")
            continue

        retrieved_chunks = hyde(query, collection_name=collection_name)

        if not retrieved_chunks:
            print("No relevant results found")
        else:
            final_ans = get_final_answer(query, retrieved_chunks)
            print(f"\nRelevant Result:\n{final_ans}")


if __name__ == "__main__":
    main()
