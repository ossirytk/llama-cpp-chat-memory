import argparse
import json
import logging
import os
from os.path import exists, join
from typing import List, Sequence

from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import AsyncChromiumLoader
from langchain.schema import Document

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
load_dotenv(find_dotenv())


def remove_unwanted_tags_and_lines(html_content: str, unwanted_tags: List[str]) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    return soup


def extract_tags(html_content: BeautifulSoup, tags: List[str], unwanted_lines: List[str]) -> str:
    text_parts: List[str] = []
    for element in html_content.find_all():
        wanted = True
        # TODO this should probably be optimized
        for line in unwanted_lines:
            ## .count('\n') to remove strings of extra lines
            ## Remove preceding or trailing \n
            if line in element.text:
                wanted = False
            if element.text.count("\n") > 1 and len(element.text) < 10:
                wanted = False

        if element.name in tags and wanted:
            text_parts.append(element.text)

    return text_parts


def clean_html(
    documents: Sequence[Document],
    tags_to_extract,
    unwanted_tags,
    unwanted_lines,
) -> Sequence[Document]:
    if tags_to_extract is None:
        tags_to_extract = ["p", "li", "div", "a"]
    if unwanted_tags is None:
        unwanted_tags = ["script", "style", "footer"]

    if unwanted_lines is None:
        unwanted_lines = ["fandom"]

    cleaned_documents = []
    logging.debug("Processing Corpus")
    for document in documents:
        cleaned_content = remove_unwanted_tags_and_lines(document.page_content, unwanted_tags)
        cleaned_content = extract_tags(cleaned_content, tags_to_extract, unwanted_lines)
        cleaned_documents.extend(cleaned_content)

    return cleaned_documents


def main(
    documents_directory: str,
    collection_name: str,
    web_scrape_directory: str,
) -> None:
    web_scrape_path = join(".", web_scrape_directory, collection_name + ".json")
    if exists(web_scrape_path):
        with open(web_scrape_path) as key_file:
            content = key_file.read()
        scrape_configs = json.loads(content)
    else:
        logging.debug("Could not load filter list")
        return
    logging.debug(f"Scraping pages: {scrape_configs['pages']}")
    logging.debug(f"Extracting tags: {scrape_configs['tags_to_extract']}")
    logging.debug(f"Unwanted tags: {scrape_configs['unwanted_tags']}")
    logging.debug(f"Unwanted lines: {scrape_configs['unwanted_lines']}")

    logging.info("Loading html")
    loader = AsyncChromiumLoader(scrape_configs["pages"])
    html = loader.load()

    logging.info("Transforming documents")
    docs_transformed = clean_html(
        html,
        scrape_configs["tags_to_extract"],
        scrape_configs["unwanted_tags"],
        scrape_configs["unwanted_lines"],
    )

    logging.info("Saving Corpus")
    storage_path = os.path.join(documents_directory, collection_name + ".txt")

    logging.info(f"Created {len(docs_transformed) } documents")
    with open(storage_path, "w", encoding="utf-8") as file:
        for doc in docs_transformed:
            file.write(doc + "\n")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="./documents/warhammer_40k",
        help="The directory where your text files are stored",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="warhammer_40k",
        help="The name of the Chroma collection",
    )

    parser.add_argument(
        "--web-scrape-directory",
        type=str,
        default="./web_scrape_configs/",
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        web_scrape_directory=args.web_scrape_directory,
    )
