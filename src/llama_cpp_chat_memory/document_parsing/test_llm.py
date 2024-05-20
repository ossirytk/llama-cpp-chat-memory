import argparse
import asyncio

from conversation_manager import ConveresationManager

conversation_manager = ConveresationManager()


async def get_answer(message):
    await conversation_manager.ask_question_test(message)


async def main(
    query: str,
) -> None:
    await get_answer(query)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--query",
        type=str,
        default="What is Polito to cyborgs and Shodan?",
        help="Query to the vector storage",
    )

    # Parse arguments
    args = parser.parse_args()

    asyncio.run(
        main(
            query=args.query,
        )
    )
