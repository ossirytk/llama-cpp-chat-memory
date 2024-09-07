from os import makedirs

import click
import pandas as pd
from conversation_manager import ConveresationManager
from langchain_core.documents.base import Document


@click.command()
@click.argument("query")
@click.option(
    "--ttype",
    "-t",
    default="mes",
    type=click.Choice(["mes", "context"]),
    help="Test type.",
)
@click.option("--keywords", "-k", default="polito, cyborgs, shodan", help="Query metadata keywords.")
@click.option("--ksize", default=3, help="The amount of context to fetch")
def main(query: str, ttype, keywords: str, ksize: int) -> None:
    """
    This script is for doing tests on embeddings. Retuns metadata results from the vector storage.
    """
    test_path = "./test/"
    # test_file = "./test/test.json"
    makedirs(test_path, exist_ok=True)
    conversation_manager = ConveresationManager(test="Testing")

    metadata_filter = conversation_manager.get_metadata_filter(keywords, ttype)
    docs: list[tuple[Document, float]] = conversation_manager.get_vector(query, ttype, metadata_filter, ksize)

    df: pd.DataFrame = conversation_manager.calculate_fusion_rank(query, docs)

    # result = df.to_json(orient="split")
    # with open(test_file, "w") as w:
    #    w.write(result)

    df = df.iloc[0:ksize]
    for line_item in df["content"].tolist():
        print(line_item)
        print("-----------")


if __name__ == "__main__":
    main()
