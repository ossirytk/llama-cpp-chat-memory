import asyncio

import click
from conversation_manager import ConveresationManager

conversation_manager = ConveresationManager()


async def test_llm(query: str):
    await conversation_manager.ask_question_test(query)


@click.command()
@click.argument("query")
def main(
    query: str,
) -> None:
    """
    This script is for doing quick tests on the model. Runs a single shot query.
    """
    asyncio.run(test_llm(query))


if __name__ == "__main__":
    main()
