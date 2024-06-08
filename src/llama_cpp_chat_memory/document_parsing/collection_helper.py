import argparse
import logging

import chromadb
from chromadb.api.client import Client
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)

load_dotenv(find_dotenv())


def main(
    collection_name: str,
    persist_directory: str,
    command: str,
) -> None:
    client: Client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    match command:
        case "list":
            logging.info("Available collections:")
            collections = client.list_collections()
            for collection in collections:
                logging.info(collection.name)
        case "delete":
            logging.info(f"Deleting {collection_name}")
            client.delete_collection(collection_name)
            logging.info(f"{collection_name} deleted")
        case _:
            collections = client.list_collections()
            logging.info("Available collections:")
            for collection in collections:
                logging.info(collection.name)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Parse text into documents and upload to chroma")

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
        "--command",
        type=str,
        default="list",
        help='Command for chroma client. "list" or "delete"',
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        command=args.command,
    )
