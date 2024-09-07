import logging

import chromadb
import click
from chromadb.api.client import Client
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)

load_dotenv(find_dotenv())


@click.command()
@click.argument("command", type=click.Choice(["list", "delete"]))
@click.option(
    "--collection-name",
    "-c",
    "collection_name",
    default="skynet",
    help="The name of the Chroma collection that's the target of an action",
)
@click.option(
    "--persist-directory",
    "-p",
    "persist_directory",
    default="./run_files/character_storage/",
    help="The directory where you want to store the Chroma collection",
)
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
    main()
