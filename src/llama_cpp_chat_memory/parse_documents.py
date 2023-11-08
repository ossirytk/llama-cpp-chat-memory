import argparse
import glob
import json
import logging
import os
from os import getenv
from os.path import join

from dotenv import find_dotenv, load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())


def main(
    documents_directory,
    collection_name,
    persist_directory,
) -> None:
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)
    params = {
        "n_ctx": getenv("N_CTX"),
        "n_batch": 1024,
        "n_gpu_layers": getenv("LAYERS"),
    }

    logging.info(f"Reading files from: {documents_directory}")
    logging.info(f"Writing to: {collection_name}")
    logging.info(f"Saving collection to: {persist_directory}")

    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths_txt = glob.glob(documents_pattern)

    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    for txt_document in documents_paths_txt:
        loader = TextLoader(txt_document, encoding="utf-8")
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        all_documents.extend(docs)

    documents_pattern = os.path.join(documents_directory, "*.json")
    documents_paths_json = glob.glob(documents_pattern)

    for json_document in documents_paths_json:
        with open(json_document) as f:
            content = f.read()
        document_content = json.loads(content)
        document_text = ""
        if isinstance(document_content["entries"], list):
            for entry in document_content["entries"]:
                if "content" in entry:
                    document_text = document_text + entry["content"]
                elif "entry" in entry:
                    document_text = document_text + entry["entry"]
            metadata = {"source": json_document}
            json_doc = [Document(page_content=document_text, metadata=metadata)]
            json_document_content = text_splitter.split_documents(json_doc)
            all_documents.extend(json_document_content)
        elif isinstance(document_content["entries"], dict):
            for entry in document_content["entries"]:
                document_text = document_text + document_content["entries"][entry]["content"]
            metadata = {"source": json_document}
            json_doc = [Document(page_content=document_text, metadata=metadata)]
            json_document_content = text_splitter.split_documents(json_doc)
            all_documents.extend(json_document_content)
    # logging.info(all_documents)
    llama = LlamaCppEmbeddings(
        model_path=model_source,
        **params,
    )
    Chroma.from_documents(
        documents=all_documents,
        embedding=llama,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )


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

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )
