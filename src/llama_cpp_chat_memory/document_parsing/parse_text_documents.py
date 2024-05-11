import argparse
import glob
import json
import logging
import multiprocessing as mp
import os

# For perf measuring
import time
from multiprocessing import Manager, Pool
from os import getenv
from os.path import join

import chromadb
import pandas as pd
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document

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


def read_documents(
    all_documents,
    que,
    reader_num,
) -> bool:
    NER_LOGGER.info("Reading documents to que")
    for doc in all_documents:
        que.put(doc)
    for _i in range(reader_num):
        que.put("QUEUE_DONE")
    NER_LOGGER.info("Reader done")
    return True


def process_documents(all_keys, read_que, write_que, name) -> bool:
    NER_LOGGER.info(f"Processor {name} reading documents from que")
    while True:
        try:
            document = read_que.get(timeout=10)
        except Exception as e:
            NER_LOGGER.info(f"Processor {name} timed out: {e}")
            write_que.put("QUEUE_DONE")
            return False

        if document == "QUEUE_DONE":
            NER_LOGGER.info(f"Processor {name} done")
            write_que.put("QUEUE_DONE")
            break

        for key in all_keys:
            if all_keys[key] in document.page_content:
                document.metadata[key] = all_keys[key]
        write_que.put(document)
    return True


def clean_and_merge_documents(que, name) -> pd.DataFrame:
    NER_LOGGER.info(f"cleaner {name} reading documents from que")
    document_list = []
    while True:
        try:
            document = que.get(timeout=10)
        except Exception as e:
            NER_LOGGER.info(f"Writer {name} timed out: {e}")
            return document_list
        if not isinstance(document, Document) and document == "QUEUE_DONE":
            NER_LOGGER.info(f"Writer {name} received done")
            break
        elif isinstance(document, Document):
            NER_LOGGER.info(f"Writer {name} received a document")
            document_list.append(document)

    return document_list


def main(
    documents_directory: str,
    collection_name: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
    key_storage: str,
    embeddings_type: str,
    threads: int,
) -> None:
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)
    embeddings_model = getenv("EMBEDDINGS_MODEL")

    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)

    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for txt_document in documents_paths_txt:
        loader = TextLoader(txt_document, encoding="utf-8")
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        all_documents.extend(docs)

    key_storage_path = join(key_storage, collection_name + ".json")

    all_keys = None
    with open(key_storage_path, encoding="utf-8") as key_file:
        content = key_file.read()
    all_keys = json.loads(content)
    if "Content" in all_keys:
        all_keys = all_keys["Content"]

    NER_LOGGER.info(f"Loading filter list from: {key_storage_path}")
    # Start Timer
    tic = time.perf_counter()

    manager = Manager()
    read_que = manager.Queue()
    write_que = manager.Queue()

    pool = Pool(threads)

    reader = pool.apply_async(
        read_documents,
        (
            all_documents,
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
            process_documents,
            (
                all_keys,
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
            clean_and_merge_documents,
            (
                write_que,
                i,
            ),
        )
        jobs.append(job)

    document_list = None
    for job in jobs:
        merge_result = job.get()
        if merge_result is not None:
            if document_list is None:
                document_list = merge_result
            else:
                document_list = document_list + merge_result

    pool.close()
    pool.join()

    # Stop timer
    toc = time.perf_counter()
    NER_LOGGER.info(f"Keys took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    if embeddings_type == "llama":
        NER_LOGGER.info("Using llama embeddigs")
        params = {
            "n_ctx": getenv("N_CTX"),
            "n_batch": 1024,
            "n_gpu_layers": getenv("LAYERS"),
        }
        embedder = LlamaCppEmbeddings(
            model_path=model_source,
            **params,
        )
    elif embeddings_type == "spacy":
        NER_LOGGER.info("Using spacy embeddigs")
        # embedder = CustomSpacyEmbeddings(model_path="en_core_web_lg")
        embedder = CustomSpacyEmbeddings(model_path=embeddings_model)
    elif embeddings_type == "huggingface":
        NER_LOGGER.info("Using huggingface embeddigs")
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embedder = HuggingFaceEmbeddings(
            model_name=embeddings_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

    else:
        error_message = f"Unsupported embeddings type: {embeddings_type}"
        raise ValueError(error_message)
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    Chroma.from_documents(
        client=client,
        documents=document_list,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Stop timer
    toc = time.perf_counter()
    NER_LOGGER.info(f"Storing embeddings took {toc - tic:0.4f} seconds")

    NER_LOGGER.info(f"Read metadata filters from directory: {key_storage_path}")
    NER_LOGGER.info(f"Read files from directory: {documents_directory}")
    NER_LOGGER.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    NER_LOGGER.info(f"Saved collection as: {collection_name}")
    NER_LOGGER.info(f"Saved collection to: {persist_directory}")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Parse text into documents and upload to chroma")

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
        "--persist-directory",
        type=str,
        default="./run_files/character_storage/",
        help="The directory where you want to store the Chroma collection",
    )

    parser.add_argument(
        "--key-storage",
        type=str,
        default="./run_files/key_storage/",
        help="The directory for the collection metadata keys",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="The text chunk size for parsing",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=1024,
        help="The overlap for text chunks for parsing",
    )
    parser.add_argument(
        "--embeddings-type",
        type=str,
        default="spacy",
        help="The chosen embeddings type",
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
        persist_directory=args.persist_directory,
        key_storage=args.key_storage,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embeddings_type=args.embeddings_type,
        threads=args.threads,
    )
