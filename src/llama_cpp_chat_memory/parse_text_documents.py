import argparse
import glob
import logging
import os
from os import getenv
from os.path import join

import chromadb
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())


def main(
    documents_directory,
    collection_name,
    persist_directory,
    chunk_size,
    chunk_overlap,
) -> None:
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)
    params = {
        "n_ctx": getenv("N_CTX"),
        "n_batch": 1024,
        "n_gpu_layers": getenv("LAYERS"),
    }

    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)

    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for txt_document in documents_paths_txt:
        loader = TextLoader(txt_document, encoding="utf-8")
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        all_documents.extend(docs)

    llama = LlamaCppEmbeddings(
        model_path=model_source,
        **params,
    )
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    Chroma.from_documents(
        client=client,
        documents=all_documents,
        embedding=llama,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )

    # If you enable this you might want to pipe the output to a file
    # logging.debug(all_documents)

    logging.info(f"Read files from directory: {documents_directory}")
    logging.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    logging.debug(f"Saved collection as: {collection_name}")
    logging.debug(f"Saved collection to: {persist_directory}")


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
        "--persist-directory",
        type=str,
        default="./character_storage/",
        help="The directory where you want to store the Chroma collection",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="The text chunk size for parsing",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=512,
        help="The overlap for text chunks for parsing",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
