import argparse
import glob
import json
import logging
import os
from os import getenv
from os.path import join

import chromadb
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma

# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())


def main(
    documents_directory: str,
    collection_name: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
    key_storage: str,
    embeddings_type: str,
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

    with open(key_storage_path, encoding="utf-8") as key_file:
        content = key_file.read()
    all_keys = json.loads(content)
    if "Content" in all_keys:
        all_keys = all_keys["Content"]

    logging.debug(f"Loading filter list from: {key_storage_path}")
    # logging.debug(f"Filter keys: {all_keys}")

    # If a metadata filter is found in the chunk, then add as metadata for that chunk
    for chunk in all_documents:
        logging.debug("-----------------------------------")
        for key in all_keys:
            if all_keys[key].lower() in chunk.page_content.lower():
                chunk.metadata[key] = all_keys[key]
        logging.debug(chunk)

    if embeddings_type == "llama":
        logging.info("Using llama embeddigs")
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
        logging.info("Using spacy embeddigs")
        # embedder = CustomSpacyEmbeddings(model_path="en_core_web_lg")
        embedder = CustomSpacyEmbeddings(model_path=embeddings_model)
    elif embeddings_type == "huggingface":
        logging.info("Using huggingface embeddigs")
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
        documents=all_documents,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )

    logging.info(f"Read metadata filters from directory: {key_storage_path}")
    logging.info(f"Read files from directory: {documents_directory}")
    logging.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    logging.debug(f"Saved collection as: {collection_name}")
    logging.debug(f"Saved collection to: {persist_directory}")


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
    )
