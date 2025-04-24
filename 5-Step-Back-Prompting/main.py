from step_back_prompting import step_back_prompting, get_final_answer


def main():
    while True:
        query = input("> ")
        if query.lower() == "clear":
            break

        retrieved_data = step_back_prompting(query)

        if not retrieved_data:
            print("No relevant results found")
        else:
            final_ans = get_final_answer(query, retrieved_data)

            print(f"\nRelevant Result:\n{final_ans}")


if __name__ == "__main__":
    main()
