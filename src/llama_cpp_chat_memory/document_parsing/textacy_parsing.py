import argparse
import glob
import json
import logging
import os
import uuid
from functools import partial
from os.path import exists

from document_parsing.extract import entities, ngrams, terms, terms_to_strings
from document_parsing.preprocessing import make_pipeline
from document_parsing.preprocessing.normalize import quotation_marks
from document_parsing.preprocessing.remove import brackets, html_tags
from document_parsing.preprocessing.replace import hashtags, urls
from document_parsing.spacier import core
from dotenv import find_dotenv, load_dotenv
from spacy.tokens import Doc

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())

SPACY_CHARACTER_LIMIT = 1000000


def process_documents(
    documents: Doc, documents_directory: str, key_storage: str, collection_name: str, write_mode: str
):
    # You can use spacy.explain to get a description for these terms
    # Or see the model in https://spacy.io/usage/models and look for model label data
    logging.debug("Extracting terms from corpus")
    extracted_terms = terms(
        documents,
        ngs=partial(ngrams, n=2, include_pos={"PROPN", "NOUN", "ADJ"}),
        ents=partial(
            entities,
            include_types={
                "PRODUCT",
                "EVENT",
                "FAC",
                "NORP",
                "PERSON",
                "ORG",
                "GPE",
                "LOC",
                "DATE",
                "TIME",
                "WORK_OF_ART",
            },
        ),
        dedupe=True,
    )
    # )

    lemma_strings = list(terms_to_strings(extracted_terms, by="lemma"))

    # Filter duplicates
    uniques = set(lemma_strings)
    all_keys = {}

    logging.debug(f"{len(uniques)} metadata keys added")

    # Create uuids for metadata filters and cleanup for entitites
    for line in uniques:
        filter_uuid = str(uuid.uuid1())
        all_keys[filter_uuid] = line

    # logging.debug(json_key_file)
    key_storage_path = os.path.join(key_storage, collection_name + ".json")

    if exists(key_storage_path):
        logging.debug("Update key file")
        with open(key_storage_path, encoding="utf-8") as key_file:
            data = json.load(key_file)

        data.update(all_keys)
        json_key_file = json.dumps(data)
        with open(key_storage_path, "w", encoding="utf-8") as key_file:
            key_file.write(json_key_file)
    else:
        logging.debug("Create key file")
        json_key_file = json.dumps(all_keys)
        with open(key_storage_path, write_mode, encoding="utf-8") as key_file:
            key_file.write(json_key_file)

    logging.info(f"Read files from directory: {documents_directory}")
    logging.info(f"Wrote keys to: {key_storage_path}")


def main(
    documents_directory: str,
    collection_name: str,
    key_storage: str,
) -> None:
    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)
    text_corpus = ""

    for txt_document in documents_paths_txt:
        logging.debug(f"Reading: {txt_document}")
        with open(txt_document, encoding="utf-8") as f:
            content = f.read()
            text_corpus = text_corpus + content

    # See https://textacy.readthedocs.io/en/latest/api_reference/preprocessing.html for options
    preproc = make_pipeline(
        quotation_marks,
        brackets,
        html_tags,
        urls,
        hashtags,
    )

    logging.debug("Cleaning Corpus")
    cleaned_corpus = preproc(text_corpus)

    # the mac corpus size is 1000000 characters so need to split the documents
    if len(cleaned_corpus) > SPACY_CHARACTER_LIMIT:
        parts = [
            cleaned_corpus[i : i + SPACY_CHARACTER_LIMIT] for i in range(0, len(cleaned_corpus), SPACY_CHARACTER_LIMIT)
        ]

        for corpus in parts:
            doc = core.make_spacy_doc(corpus, lang="en_core_web_lg")
            process_documents(doc, documents_directory, key_storage, collection_name, "a")
    else:
        # See https://spacy.io/usage/models for options
        doc = core.make_spacy_doc(cleaned_corpus, lang="en_core_web_lg")
        process_documents(doc, documents_directory, key_storage, collection_name, "w")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="./documents/skynet",
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
        default="./key_storage/",
        help="The directory where you want to store the Chroma collection metadata keys",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        key_storage=args.key_storage,
    )
