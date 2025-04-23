from reciprocal_rank_fusion import reciprocal_rank_fusion, get_final_answer


def main():
    while True:
        query = input("> ")
        if query.lower() == "clear":
            break

        retrieved_chunks = reciprocal_rank_fusion(query)

        if not retrieved_chunks:
            print("No relevant results found")
        else:
            final_ans = get_final_answer(query, retrieved_chunks)

            print(f"\nRelevant Result:\n{final_ans}")


if __name__ == "__main__":
    main()
