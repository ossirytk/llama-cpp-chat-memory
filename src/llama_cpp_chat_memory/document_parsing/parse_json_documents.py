import argparse
import glob
import json
import logging
import os
import uuid
from os import getenv
from os.path import join

import chromadb
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain.docstore.document import Document
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
    key_storage,
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

    all_documents = []
    all_keys = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents_pattern = os.path.join(documents_directory, "*.json")
    documents_paths_json = glob.glob(documents_pattern)

    for json_document in documents_paths_json:
        with open(json_document) as f:
            content = f.read()
        document_content = json.loads(content)
        if isinstance(document_content["entries"], list):
            logging.debug("Parsing List")
            for entry in document_content["entries"]:
                document_text = ""
                metadata_filters = {"source": json_document}

                if "content" in entry:
                    document_text = document_text + entry["content"]
                elif "entry" in entry:
                    document_text = document_text + entry["entry"]

                logging.debug(f"Extracted a key: {entry['keys']}")
                for m_filter in entry["keys"]:
                    filter_uuid = str(uuid.uuid1())
                    metadata_filters[filter_uuid] = m_filter

                all_keys = metadata_filters
                json_doc = [Document(page_content=document_text, metadata=metadata_filters)]
                json_document_content = text_splitter.split_documents(json_doc)
                all_documents.extend(json_document_content)
        elif isinstance(document_content["entries"], dict):
            logging.debug("Parsing dict")
            for entry in document_content["entries"]:
                metadata_filters = {"source": json_document}
                document_text = document_text + document_content["entries"][entry]["content"]

                logging.debug(f"Extracted a key: {document_content['entries'][entry]['key']}")
                for m_filter in document_content["entries"][entry]["key"]:
                    filter_uuid = str(uuid.uuid1())
                    metadata_filters[filter_uuid] = m_filter

                all_keys = metadata_filters
                json_doc = [Document(page_content=document_text, metadata=metadata_filters)]
                json_document_content = text_splitter.split_documents(json_doc)
                all_documents.extend(json_document_content)

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

    logging.debug(f"Key file content: {all_keys}")

    json_key_file = json.dumps(all_keys)
    # logging.debug(f"Key file uuid keys: {list(all_keys.keys())}")

    # If you enable this you might want to pipe the output to a file
    # logging.debug(all_documents)

    key_storage_path = os.path.join(key_storage, collection_name + ".json")
    with open(key_storage_path, "w") as key_file:
        key_file.write(json_key_file)

    logging.info(f"Read files from directory: {documents_directory}")
    logging.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    logging.debug(f"Saved collection as: {collection_name}")
    logging.debug(f"Saved collection to: {persist_directory}")
    logging.info(f"Wrote keys to: {key_storage_path}")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="../documents/hogwarts",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="hogwarts",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist-directory",
        type=str,
        default="../character_storage/",
        help="The directory where you want to store the Chroma collection",
    )

    parser.add_argument(
        "--key-storage",
        type=str,
        default="../key_storage/",
        help="The directory where you want to store the Chroma collection metadata keys",
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

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        key_storage=args.key_storage,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
