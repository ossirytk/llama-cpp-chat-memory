import argparse

from conversation_manager import ConveresationManager


def main(
    query: str,
) -> None:
    # TODO test builder for manager so you can set the params here
    conversation_manager = ConveresationManager()
    vector_context = conversation_manager.get_vector(query, "context")
    print(vector_context)

    mes_context = conversation_manager.get_vector(query, "mes")
    print(mes_context)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Test chroma embeddings")

    # Add arguments
    parser.add_argument(
        "--query",
        type=str,
        default="What is Polito to cyborgs and Shodan?",
        help="Query to the vector storage",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        query=args.query,
    )
