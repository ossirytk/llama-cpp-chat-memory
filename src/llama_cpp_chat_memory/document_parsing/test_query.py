import json
import logging
from functools import partial
from os.path import exists, join

import click
import spacy
from dotenv import find_dotenv, load_dotenv

from document_parsing.extract import entities, ngrams, terms
from document_parsing.extract.basics import terms_to_strings

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())


@click.command()
@click.argument("query")
@click.option(
    "--model",
    "-m",
    default="en_core_web_lg",
    help="The spacy model to parse the text",
)
@click.option(
    "--parse-config-directory", "-pcd", default="./run_files/parse_configs/", help="The parse config directory"
)
@click.option(
    "--parse-config-file",
    "-pcf",
    default="query_metadata_filter.json",
    help="The parse config file",
)
def main(
    query: str,
    model: str,
    parse_config_directory: str,
    parse_config_file: str,
) -> None:
    """
    This script is for testing metadata parsing with spacy. Parses the keywords from a query.
    """
    spacy_lang = spacy.load(model)
    doc = spacy_lang(query)
    parse_config_path = join(".", parse_config_directory, parse_config_file)
    if exists(parse_config_path):
        with open(parse_config_path) as key_file:
            filter_content = key_file.read()
        filter_configs = json.loads(filter_content)
    else:
        logging.info("Could not load parse config file")
        return

    ngrams_list = filter_configs["ngs"]
    entities_list = filter_configs["entities"]
    noun_chunks = filter_configs["noun_chunks"]
    extract_type = filter_configs["extract_type"]

    logging.info("Extracting terms from corpus")
    extracted_terms = terms(
        doc,
        ngs=partial(ngrams, n=noun_chunks, include_pos=ngrams_list),
        ents=partial(
            entities,
            include_types=entities_list,
        ),
        dedupe=True,
    )

    lemma_strings = list(terms_to_strings(extracted_terms, by=extract_type))
    logging.info(lemma_strings)


if __name__ == "__main__":
    main()
