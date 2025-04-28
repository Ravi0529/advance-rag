from HyDE import hyde, get_final_answer
from router import semantic_routing


def main():
    while True:
        query = input("> ")
        if query.lower() == "clear":
            break

        collection_name = semantic_routing(
            query
        )  # basically, let the AI model decide which collection you should use for answering the query(semantic routing)
        # print(collection_name)

        if collection_name is None:
            print("Sorry, couldn't classify your query into Node.js or Python.")
            continue

        retrieved_chunks = hyde(query, collection_name=collection_name)

        if not retrieved_chunks:
            print("No relevant results found.")
            continue

        final_ans = get_final_answer(query, retrieved_chunks)
        print(f"\nRelevant Result:\n{final_ans}")


if __name__ == "__main__":
    main()
