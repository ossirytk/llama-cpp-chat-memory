import argparse
import glob
import logging
import os
from os import getenv
from os.path import join

import chromadb
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())


def main(
    documents_directory: str,
    collection_name: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings_type: str,
) -> None:
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)
    embeddings_model = getenv("EMBEDDINGS_MODEL")

    documents_pattern = os.path.join(documents_directory, "*.pdf")
    logging.debug(f"documents search pattern: {documents_pattern}")
    documents_paths_pdf = glob.glob(documents_pattern)

    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for pdf_document in documents_paths_pdf:
        logging.debug(f"loading: {pdf_document}")
        loader = PyPDFLoader(pdf_document)
        docs = loader.load_and_split(text_splitter=text_splitter)
        all_documents.extend(docs)

    if embeddings_type == "llama":
        params = {
            "n_ctx": getenv("N_CTX"),
            "n_batch": 1024,
            "n_gpu_layers": getenv("LAYERS"),
        }

        logging.info("Using llama embeddigs")
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

    # If you enable this you might want to pipe the output to a file
    # logging.debug(all_documents)

    logging.info(f"Read files from directory: {documents_directory}")
    logging.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    logging.debug(f"Saved collection as: {collection_name}")
    logging.debug(f"Saved collection to: {persist_directory}")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Parse pdf documents to documents and upload to chroma")

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="./run_files/documents/fyodor_dostoyevsky",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="dostoyevsky",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist-directory",
        type=str,
        default="./run_files/character_storage/",
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
        default=0,
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
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embeddings_type=args.embeddings_type,
    )
