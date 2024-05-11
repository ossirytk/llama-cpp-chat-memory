import argparse
import json
import logging
import os
import re
from os.path import exists, join

from dotenv import find_dotenv, load_dotenv
from trafilatura import extract, fetch_url

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())


def main(
    documents_directory: str,
    collection_name: str,
    web_scrape_directory: str,
    filter_directory: str,
    filter_file: str,
) -> None:
    web_scrape_path = join(".", web_scrape_directory, collection_name + ".json")
    if exists(web_scrape_path):
        with open(web_scrape_path) as key_file:
            content = key_file.read()
        scrape_configs = json.loads(content)
    else:
        logging.debug("Could not load filter list")
        return

    filters_path = join(".", filter_directory, filter_file)
    if exists(filters_path):
        with open(filters_path) as key_file:
            filter_content = key_file.read()
        filter_configs = json.loads(filter_content)
    else:
        logging.debug("Could not load filter list")
        return

    parse_filters = filter_configs["filters"]

    storage_path = os.path.join(documents_directory, collection_name + ".txt")
    for page in scrape_configs["pages"]:
        logging.info("Loading html")
        downloaded = fetch_url(page)

        if downloaded is not None:
            logging.info("Transforming documents")
            result = extract(
                downloaded, include_comments=False, include_images=False, include_links=False, include_tables=False
            )

            for parse_filter in parse_filters:
                filter_iterator = iter(parse_filter)
                parse_regex = next(filter_iterator)
                parse_replacment = next(filter_iterator)
                result = re.sub(parse_filter[parse_regex], parse_filter[parse_replacment], result)

            logging.info("Saving Corpus")
            with open(storage_path, "a", encoding="utf-8") as file:
                file.write(result + "\n")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Web scrape web pages into text")

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
        help="The name of the collection. Should match eventual Choma collection",
    )

    parser.add_argument(
        "--web-scrape-directory",
        type=str,
        default="./run_files/web_scrape_configs/",
        help="The config file to be used for the webscrape",
    )

    parser.add_argument(
        "--filter-directory",
        type=str,
        default="./run_files/filters/",
        help="The filter directory",
    )

    parser.add_argument(
        "--filter-file",
        type=str,
        default="web_scrape_filter.json",
        help="The web scrape filter",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        web_scrape_directory=args.web_scrape_directory,
        filter_directory=args.filter_directory,
        filter_file=args.filter_file,
    )
