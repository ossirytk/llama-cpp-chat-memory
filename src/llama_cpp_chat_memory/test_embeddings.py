import argparse
import json
import logging
from os import getenv
from os.path import join

import chromadb
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())


def main(
    query,
    k,
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

    llama = LlamaCppEmbeddings(
        model_path=model_source,
        **params,
    )
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    db = Chroma(
        client=client, collection_name=collection_name, persist_directory=persist_directory, embedding_function=llama
    )

    use_keys = getenv("USE_KEY_STORAGE")
    all_keys = None
    if use_keys:
        key_storage = getenv("KEY_STORAGE_DIRECTORY")
        key_storage_path = join(key_storage, collection_name + ".json")
        with open(key_storage_path) as key_file:
            content = key_file.read()
        all_keys = json.loads(content)
    logging.debug(f"Loading filter list from: {key_storage_path}")
    logging.debug(f"Filter keys: {all_keys}")

    # Currently Chroma has no "like" implementation so this is a case sensitive hack
    # There is also an issue with the filter only having one item so we use filter_list in this case
    metadata_filter_list = []
    filter_list = {}
    for item in all_keys.items():
        if item[1] in query:
            filter_list[item[0]] = item[1]
            metadata_filter_list.append({item[0]: {"$in": [item[1]]}})

    logging.info(f"There are {db._collection.count()} documents in the collection")
    logging.info("Similiarity search")
    logging.info("--------------------------------------------------------------------")
    if len(filter_list) == 1:
        where = filter_list
    elif len(filter_list) > 1:
        where = {"$or": metadata_filter_list}
    else:
        where = None
    # query it
    docs = db.similarity_search_with_score(query=query, k=k, filter=where)
    vector_context = ""
    for answer in docs:
        logging.info("--------------------------------------------------------------------")
        logging.info(f"distance: {answer[1]}")
        logging.info(answer[0].metadata)
        logging.info(answer[0].page_content)
        vector_context = vector_context + answer[0].page_content
    logging.info("--------------------------------------------------------------------")
    logging.info(vector_context)

    logging.info("Max marginal relevance search")
    logging.info("--------------------------------------------------------------------")
    docs = db.max_marginal_relevance_search(query=query, k=k, fetch_k=10, lambda_mult=0.75, filter=where)

    vector_context = ""
    for answer in docs:
        logging.info("--------------------------------------------------------------------")
        logging.info(answer.page_content)
        vector_context = vector_context + answer.page_content
    logging.info("--------------------------------------------------------------------")
    logging.info(vector_context)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--query",
        type=str,
        default="Who is John Connor?",
        help="Query to the vector storage",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="The nuber of documents to fetch",
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
        query=args.query,
        k=args.k,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )
