import argparse
import glob
import json
import logging
import os
import uuid
from functools import partial

import textacy
from dotenv import find_dotenv, load_dotenv
from textacy import extract, preprocessing

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())


def main(
    documents_directory,
    collection_name,
    key_storage,
) -> None:
    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)
    text_corpus = ""

    for txt_document in documents_paths_txt:
        with open(txt_document) as f:
            content = f.read()
            text_corpus = text_corpus + content

    preproc = preprocessing.make_pipeline(
        preprocessing.remove.punctuation,
        preprocessing.normalize.whitespace,
        preprocessing.normalize.hyphenated_words,
    )

    cleaned_corpus = preproc(text_corpus)
    doc = textacy.make_spacy_doc(cleaned_corpus, lang="en_core_web_lg")

    terms = list(
        extract.terms(
            doc,
            ents=partial(extract.entities, include_types={"PERSON", "ORG", "GPE", "LOC"}),
            dedupe=True,
        )
    )

    logging.debug(terms)
    logging.debug("---------------------------------------------------------------")
    # Filter duplicates
    uniques = {term.text for term in terms}
    all_keys = {}
    # Create uuids for metadata filters and cleanup for entitites
    for line in uniques:
        if "\n" not in line and " s" not in line:
            filter_uuid = str(uuid.uuid1())
            all_keys[filter_uuid] = line

    json_key_file = json.dumps(all_keys)

    key_storage_path = os.path.join(key_storage, collection_name + ".json")
    with open(key_storage_path, "w") as key_file:
        key_file.write(json_key_file)

    logging.info(f"Read files from directory: {documents_directory}")
    logging.info(f"Wrote keys to: {key_storage_path}")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="../documents/skynet",
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
        default="../key_storage/",
        help="The directory where you want to store the Chroma collection metadata keys",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        key_storage=args.key_storage,
    )
