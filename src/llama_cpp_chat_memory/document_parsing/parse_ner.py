import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import uuid
from collections.abc import Iterable
from functools import partial
from multiprocessing import Manager, Pool
from os.path import exists, join

import pandas as pd
from document_parsing.extract import entities, ngrams, terms, terms_to_strings
from document_parsing.spacier import core
from dotenv import find_dotenv, load_dotenv
from spacy.tokens import Doc

# This is the config for multiprocess logger
# Setting the level to debug outputs multiprocess debug lines too
NER_LOGGER = mp.get_logger()
FORMAT = "%(levelname)s:%(message)s"
formatter = logging.Formatter(fmt=FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

NER_LOGGER.addHandler(handler)
NER_LOGGER.setLevel(logging.INFO)

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
                NER_LOGGER.warning(f"Created a chunk of size {total}, which is longer than the specified {chunk_size}")
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
) -> pd.Series:
    # You can use spacy.explain to get a description for these terms
    # Or see the model in https://spacy.io/usage/models and look for model label data

    parse_config_path = join(".", parse_config_directory, parse_config_file)
    if exists(parse_config_path):
        with open(parse_config_path) as key_file:
            filter_content = key_file.read()
        filter_configs = json.loads(filter_content)
    else:
        NER_LOGGER.info("Could not load parse config file")
        return

    ngrams_list = filter_configs["ngs"]
    entities_list = filter_configs["entities"]
    noun_chunks = filter_configs["noun_chunks"]
    extract_type = filter_configs["extract_type"]

    NER_LOGGER.info("Extracting terms from corpus")
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
    all_keys = {}

    NER_LOGGER.info(f"{len(lemma_strings)} metadata keys created")

    # Create uuids for metadata filters
    for line in lemma_strings:
        filter_uuid = str(uuid.uuid1())
        all_keys[filter_uuid] = line
    return pd.Series(all_keys)


def read_chuncks(text_corpus, chunk_size, chunk_overlap, que, reader_num) -> bool:
    NER_LOGGER.info("Reading chuncks to que")
    parts = split_text(text_corpus, chunk_size, chunk_overlap)
    for doc in parts:
        que.put(doc)
    for _i in range(reader_num):
        que.put("QUEUE_DONE")
    NER_LOGGER.info("Reader done")
    return True


def process_chuncks(model, parse_config_directory, parse_config_file, read_que, write_que, name) -> bool:
    NER_LOGGER.info(f"Processor {name} reading chuncks from que")
    while True:
        try:
            corpus = read_que.get(timeout=10)
        except Exception as e:
            NER_LOGGER.info(f"Processor {name} timed out: {e}")
            write_que.put("QUEUE_DONE")
            return False

        if corpus == "QUEUE_DONE":
            NER_LOGGER.info(f"Processor {name} done")
            write_que.put("QUEUE_DONE")
            break
        doc = core.make_spacy_doc(corpus, lang=model)
        pseries = process_documents(doc, parse_config_directory, parse_config_file)
        write_que.put(pseries)
    return True


def clean_and_merge_chunks(que, name) -> pd.DataFrame:
    NER_LOGGER.info(f"cleaner {name} reading chuncks from que")
    df = None
    while True:
        try:
            corpus = que.get(timeout=10)
        except Exception as e:
            NER_LOGGER.info(f"Writer {name} timed out: {e}")
            df["Content"].apply(lambda x: x.strip())
            # TODO Place this filter in config file
            # Removes one and two letter words
            m = ~df.apply(lambda x: x.str.contains("\\b[a-zA-Z]{1,2}\\b")).any(axis=1)
            df = df[m]
            return df
        if not isinstance(corpus, pd.Series) and corpus == "QUEUE_DONE":
            NER_LOGGER.info(f"Writer {name} received done")
            break
        elif isinstance(corpus, pd.Series):
            NER_LOGGER.info(f"Writer {name} received a chunck")
            if df is None:
                df = pd.DataFrame(corpus, columns=["Content"])
            else:
                df2 = pd.DataFrame(corpus, columns=["Content"])
                df = pd.concat([df, df2])

    df["Content"].apply(lambda x: x.strip())
    # TODO Place this filter in config file
    # Removes one and two letter words
    m = ~df.apply(lambda x: x.str.contains("\\b[a-zA-Z]{1,2}\\b")).any(axis=1)
    df = df[m]
    NER_LOGGER.info(f"writer {name} - Total amount of keys created: {len(df.index)}")

    return df


def main(
    documents_directory: str,
    collection_name: str,
    key_storage: str,
    model: str,
    parse_config_directory: str,
    parse_config_file: str,
    chunk_size: int,
    chunk_overlap: int,
    threads: int,
) -> None:
    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)
    text_corpus = ""

    for txt_document in documents_paths_txt:
        NER_LOGGER.info(f"Reading: {txt_document}")
        with open(txt_document, encoding="utf-8") as f:
            content = f.read()
            text_corpus = text_corpus + content

    manager = Manager()
    read_que = manager.Queue()
    write_que = manager.Queue()

    pool = Pool(threads)

    reader = pool.apply_async(
        read_chuncks,
        (
            text_corpus,
            chunk_size,
            chunk_overlap,
            read_que,
            threads,
        ),
    )

    read_success = reader.get()
    if not read_success:
        return

    jobs = []
    for i in range(threads):
        job = pool.apply_async(
            process_chuncks,
            (
                model,
                parse_config_directory,
                parse_config_file,
                read_que,
                write_que,
                i,
            ),
        )
        jobs.append(job)

    for job in jobs:
        job.get()

    jobs = []
    for i in range(threads):
        job = pool.apply_async(
            clean_and_merge_chunks,
            (
                write_que,
                i,
            ),
        )
        jobs.append(job)

    df = None
    for job in jobs:
        merge_result = job.get()
        if merge_result is not None:
            if df is None:
                df = merge_result
            else:
                df = pd.concat([df, merge_result])

    pool.close()
    pool.join()

    if df is not None:
        df = df.drop_duplicates()
        NER_LOGGER.info(f"Total amount of keys created: {len(df.index)}")
        key_storage_path = os.path.join(key_storage, collection_name + ".json")

        NER_LOGGER.debug("Create key file")
        json_key_file = df.to_json()
        with open(key_storage_path, mode="w", encoding="utf-8") as key_file:
            key_file.write(json_key_file)

        NER_LOGGER.info(f"Read files from directory: {documents_directory}")
        NER_LOGGER.info(f"Wrote keys to: {key_storage_path}")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

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
        default=1000000,
        help="The text chunk size for parsing. Default spacy maximum chunk size",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="The overlap for text chunks for parsing",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="The number of threads to use.",
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
        threads=args.threads,
    )
