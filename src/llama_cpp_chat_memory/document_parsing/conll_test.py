# import spacy_stanza
# import stanza
import argparse
import glob
import logging
import os

from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from spacy_conll import init_parser

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())


def main(
    documents_directory: str,
    collection_name: str,
    key_storage: str,
    model: str,
    parse_config_directory: str,
    parse_config_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)
    corpus = ""

    for txt_document in documents_paths_txt:
        logging.debug(f"Reading: {txt_document}")
        with open(txt_document, encoding="utf-8") as f:
            content = f.read()
            corpus = corpus + content

    nlp = init_parser("en", "stanza", is_tokenized=True, parser_opts={"download_method": None})

    # df = None
    doc = nlp(corpus)
    df: DataFrame = doc._.conll_pd
    logging.info(df.head())
    key_storage_path = os.path.join(key_storage, "conll_test.json")

    csv_key_file = df.to_json()
    with open(key_storage_path, mode="w", encoding="utf-8") as key_file:
        key_file.write(csv_key_file)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Parse ner keywords from text using spacy and grammar configuration files."
    )

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="./run_files/documents/skynet",
        help="The directory where your text files are stored",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="skynet",
        help="The name of the Chroma collection",
    )

    parser.add_argument(
        "--key-storage",
        type=str,
        default="./run_files/key_storage/",
        help="The directory where you want to store the Chroma collection metadata keys",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_lg",
        help="The spacy model to parse the text.",
    )

    parser.add_argument(
        "--parse-config-directory",
        type=str,
        default="./run_files/parse_configs/",
        help="The parse config directory",
    )

    parser.add_argument(
        "--parse-config-file",
        type=str,
        default="ner_types.json",
        help="The parse config file",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=524288,
        help="The text chunk size for parsing.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="The overlap for text chunks for parsing",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        key_storage=args.key_storage,
        model=args.model,
        parse_config_directory=args.parse_config_directory,
        parse_config_file=args.parse_config_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
