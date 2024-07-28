import argparse
from os import makedirs

import pandas as pd
from conversation_manager import ConveresationManager
from langchain_core.documents.base import Document


def main(query: str, ttype, keywords: str, ksize: int) -> None:
    test_path = "./test/"
    test_file = "./test/test.json"
    makedirs(test_path, exist_ok=True)
    conversation_manager = ConveresationManager(test="Testing")

    if ttype == "mes":
        retriever = conversation_manager.mes_retriever
    else:
        retriever = conversation_manager.context_retriever
    metadata_filter = conversation_manager.get_metadata_filter(keywords, conversation_manager.all_mes_keys, retriever)
    docs: list[tuple[Document, float]] = conversation_manager.get_vector(query, retriever, metadata_filter, ksize)

    df: pd.DataFrame = conversation_manager.calculate_fusion_rank(query, docs)

    result = df.to_json(orient="split")
    with open(test_file, "w") as w:
        w.write(result)

    df = df.iloc[0:ksize]
    content_list = "\n".join(df["content"].tolist())
    print(content_list)


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

    parser.add_argument(
        "--ttype",
        type=str,
        default="mes",
        help="Test type. context or mes. Default mes",
    )

    parser.add_argument(
        "--keywords",
        type=str,
        default="polito, cyborgs, shodan",
        help="Query metadata keywords",
    )

    parser.add_argument(
        "--ksize",
        type=int,
        default=1,
        help="The amount of context to fetch",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        query=args.query,
        ttype=args.ttype,
        keywords=args.keywords,
        ksize=args.ksize,
    )
