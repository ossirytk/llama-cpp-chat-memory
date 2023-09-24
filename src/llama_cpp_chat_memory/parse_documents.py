import os
import glob
from os import getenv, mkdir
from os.path import exists, join, splitext, realpath, dirname
from dotenv import load_dotenv, find_dotenv
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings

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

    print(documents_directory)
    print(collection_name)
    print(persist_directory)

    documents_pattern = os.path.join(documents_directory, "*.txt")
    documents_paths = glob.glob(documents_pattern)

    all_documents = list()
    for document_path in documents_paths:
        print(document_path)

        loader = TextLoader(document_path, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=512)
        docs = text_splitter.split_documents(documents)
        all_documents.extend(docs)

    # print(all_documents)
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
        "--data_directory",
        type=str,
        default="./documents/",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="skynet",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_directory",
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
