from parallel_query import parallel_query


def main():
    while True:
        query = input("> ")
        if query.lower() == "clear":
            break

        results = parallel_query(query)

        if not results:
            print("No relevant results found")
        else:
            print("\nRelevant Results:\n")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res}\n")


if __name__ == "__main__":
    main()
