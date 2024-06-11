# import spacy_stanza
# import stanza
import argparse
import glob
import logging
import os
import re
from collections.abc import Iterable

from dotenv import find_dotenv, load_dotenv
from spacy_conll import init_parser

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())

SPACY_CHARACTER_LIMIT = 1000000

# stanza.download("en")


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
    text_corpus = ""

    for txt_document in documents_paths_txt:
        logging.debug(f"Reading: {txt_document}")
        with open(txt_document, encoding="utf-8") as f:
            content = f.read()
            text_corpus = text_corpus + content

    logging.debug("Cleaning Corpus")

    # the max corpus size is 1000000 characters so need to split the documents
    parts = split_text(text_corpus, chunk_size, chunk_overlap)

    nlp = init_parser("en", "stanza", is_tokenized=True)

    df = None
    for corpus in parts:
        doc = nlp(corpus)
        logging.info(doc._.conll_pd.head())


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
