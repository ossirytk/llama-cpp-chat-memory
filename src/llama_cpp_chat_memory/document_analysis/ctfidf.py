import glob
import json
import logging
import math
import os
import re
from collections.abc import Iterable
from functools import partial
from os.path import exists, join
from pathlib import Path

import click
from document_parsing.extract import entities, ngrams, terms
from document_parsing.extract.basics import terms_to_strings
from document_parsing.spacier import core
from dotenv import find_dotenv, load_dotenv
from spacy.tokens import Doc

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)

load_dotenv(find_dotenv())

SPACY_CHARACTER_LIMIT = 1000000


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    separators = ["\n\n", "\n", " ", ""]

    """Split incoming text and return chunks."""
    final_chunks = []
    # Get appropriate separator to use
    separator = separators[-1]
    new_separators = []
    for i, _s in enumerate(separators):
        _separator = re.escape(_s)
        if _s == "":
            separator = _s
            break
        if re.search(_separator, text):
            separator = _s
            new_separators = separators[i + 1 :]
            break

    _separator = re.escape(separator)
    splits = split_text_with_regex(text, _separator)

    # Now go merging things, recursively splitting longer texts.
    _good_splits = []
    _separator = ""
    for s in splits:
        if _good_splits:
            merged_text = merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
            _good_splits = []
        if not new_separators:
            final_chunks.append(s)
        else:
            other_info = _split_text(s, new_separators, chunk_size, chunk_overlap)
            final_chunks.extend(other_info)
    if _good_splits:
        merged_text = merge_splits(_good_splits, _separator)
        final_chunks.extend(merged_text)
    return final_chunks


def _split_text(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split incoming text and return chunks."""
    final_chunks = []
    # Get appropriate separator to use
    separator = separators[-1]
    new_separators = []
    for i, _s in enumerate(separators):
        _separator = re.escape(_s)
        if _s == "":
            separator = _s
            break
        if re.search(_separator, text):
            separator = _s
            new_separators = separators[i + 1 :]
            break

    _separator = re.escape(separator)
    splits = split_text_with_regex(text, _separator)

    # Now go merging things, recursively splitting longer texts.
    _good_splits = []
    _separator = separator
    for s in splits:
        if len(s) < chunk_size:
            _good_splits.append(s)
        else:
            if _good_splits:
                merged_text = merge_splits(_good_splits, _separator, chunk_size, chunk_overlap)
                final_chunks.extend(merged_text)
                _good_splits = []
            if not new_separators:
                final_chunks.append(s)
            else:
                other_info = _split_text(s, new_separators, chunk_size, chunk_overlap)
                final_chunks.extend(other_info)
    if _good_splits:
        merged_text = merge_splits(_good_splits, _separator, chunk_size, chunk_overlap)
        final_chunks.extend(merged_text)
    return final_chunks


def split_text_with_regex(text: str, separator: str) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        # The parentheses in the pattern keep the delimiters in the result.
        _splits = re.split(f"({separator})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
        if len(_splits) % 2 == 0:
            splits += _splits[-1:]
        splits = [_splits[0], *splits]
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


def merge_splits(splits: Iterable[str], separator: str, chunk_size, chunk_overlap) -> list[str]:
    # We now want to combine these smaller pieces into medium size
    # chunks to send to the LLM.
    separator_len = len(separator)

    docs = []
    current_doc: list[str] = []
    total = 0
    for d in splits:
        _len = len(d)
        if total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
            if total > chunk_size:
                logging.warning(f"Created a chunk of size {total}, which is longer than the specified {chunk_size}")
            if len(current_doc) > 0:
                doc = join_docs(current_doc, separator)
                if doc is not None:
                    docs.append(doc)
                # Keep on popping if:
                # - we have a larger chunk than in the chunk overlap
                # - or if we still have any chunks and the length is long
                while total > chunk_overlap or (
                    total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size and total > 0
                ):
                    total -= len(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += _len + (separator_len if len(current_doc) > 1 else 0)
    doc = join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs


def join_docs(docs: list[str], separator: str) -> str | None:
    text = separator.join(docs)
    text = text.strip()

    if text == "":
        return None
    else:
        return text


def process_documents(
    documents: Doc,
    parse_config_directory: str,
    parse_config_file: str,
) -> list:
    # You can use spacy.explain to get a description for these terms
    # Or see the model in https://spacy.io/usage/models and look for model label data

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

    logging.debug("Extracting terms from corpus")
    extracted_terms = terms(
        documents,
        ngs=partial(ngrams, n=noun_chunks, include_pos=ngrams_list),
        ents=partial(
            entities,
            include_types=entities_list,
        ),
        dedupe=True,
    )

    lemma_strings = list(terms_to_strings(extracted_terms, by=extract_type))

    logging.debug(f"{len(lemma_strings)} metadata keys created")
    return lemma_strings


@click.command()
@click.option(
    "--documents-directory",
    "-d",
    "documents_directory",
    default="./run_files/documents/skynet",
    help="The directory where your text files are stored",
)
@click.option(
    "--key-storage", "-k", default="./run_files/key_storage/", help="The directory for the collection metadata keys."
)
@click.option(
    "--keyfile-name",
    "-k",
    "keyfile_name",
    default="keyfile",
    help="Keyfile name.",
)
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
    default="ner_types_analyze.json",
    help="The parse config file",
)
@click.option(
    "--chunk-size",
    "-cs",
    "chunk_size",
    type=int,
    default=1000000,
    help="The text chunk size for parsing. Default spacy maximum chunk size",
)
@click.option(
    "--chunk-overlap",
    "-co",
    "chunk_overlap",
    default=0,
    type=int,
    help="The overlap for text chunks for parsing",
)
def main(
    documents_directory: str,
    key_storage: str,
    keyfile_name: str,
    model: str,
    parse_config_directory: str,
    parse_config_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """This script is a rough implementation of Class-based TF-IDF.
    See: https://www.maartengrootendorst.com/blog/ctfidf/
    Tries to find words that represent a topic. Needs a large amount of topics to work.
    This is just a crude proof of concept. Has poor performance and some hacks.
    """
    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)

    # Contains the total occurances of words in all material
    all_words = {}
    # Contains a the occurance frequency in the doc and c-TF-IDF value
    all_docs = {}
    # Total word counts in docs
    total_word_count = {}
    for txt_document in documents_paths_txt:
        logging.info(f"Parsing: {txt_document}")
        with open(txt_document, encoding="utf-8") as f:
            content = f.read()
        parts = split_text(content, chunk_size, chunk_overlap)
        data = {}
        for part in parts:
            doc = core.make_spacy_doc(part, lang=model)
            words = process_documents(doc, parse_config_directory, parse_config_file)

            for word in words:
                # TODO Better word filter
                if word not in [
                    "User",
                    "user",
                ]:
                    if word in data.keys():
                        data[word] = data[word] + 1
                    else:
                        data[word] = 1
                    if word in all_words.keys():
                        all_words[word] = all_words[word] + 1
                    else:
                        all_words[word] = 1
        all_docs[txt_document] = data

        frequency = 0
        for _key, word in data.items():
            frequency = frequency + word
        total_word_count[txt_document] = frequency

    total_documents = len(all_docs)
    # TODO Process documents in threads?
    for doc_name, adoc in all_docs.items():
        logging.info(f"Words for document - {doc_name}: {len(adoc)}")
        logging.info(f"Calculating values for: {doc_name}")
        topic_collection = {}
        for word_key, word_count in adoc.items():
            word_count_in_class = word_count
            words_in_class_total = total_word_count[doc_name]
            word_count_in_total = all_words[word_key]
            log_part = math.log(total_documents / word_count_in_total)
            log_part = max(log_part, 1)
            reference_value = (word_count_in_class / words_in_class_total * log_part) * 100000
            reference_tuple = (word_count, reference_value)
            topic_collection[word_key] = reference_tuple

        topic_collection = dict(sorted(topic_collection.items(), key=lambda item: item[1][1], reverse=True))

        collection_name = Path(doc_name).stem
        complete_name = collection_name + "_" + keyfile_name
        key_storage_path = os.path.join(key_storage, complete_name + ".json")
        logging.info(f"Saving values for: {doc_name}")
        with open(key_storage_path, mode="w", encoding="utf-8") as key_file:
            json.dump(topic_collection, key_file)


if __name__ == "__main__":
    main()
