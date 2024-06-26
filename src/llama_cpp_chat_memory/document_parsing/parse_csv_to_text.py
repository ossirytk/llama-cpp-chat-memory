import argparse
import glob
import json
import logging
import os
from os.path import exists, join, splitext

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from trafilatura import extract

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())


def main(
    documents_directory: str,
    parse_config_directory: str,
    parse_config_file: str,
    filter_config_directory: str,
    filter_config_file: str,
) -> None:
    documents_pattern = os.path.join(documents_directory, "*.csv")
    logging.debug(f"documents search pattern: {documents_pattern}")
    documents_paths_csv = glob.glob(documents_pattern)

    logging.debug("Reading config file")
    parse_config_path = join(".", parse_config_directory, parse_config_file)
    if exists(parse_config_path):
        with open(parse_config_path) as key_file:
            column_content = key_file.read()
        column_configs = json.loads(column_content)
    else:
        logging.debug("Could not load parse config file")
        return

    logging.debug("Reading filter file")
    filter_config_path = join(".", filter_config_directory, filter_config_file)
    if exists(filter_config_path):
        with open(filter_config_path) as key_file:
            filter_content = key_file.read()
        filter_configs = json.loads(filter_content)
    else:
        logging.debug("Could not load parse config file")
        return

    parse_filters = filter_configs["filters"]

    for csv_document in documents_paths_csv:
        logging.debug(f"Processing: {csv_document}")
        data = ""
        columns_list = []
        with open(csv_document, encoding="utf8") as f:
            first_line = f.readline()
        columns_line = "index" + first_line.strip()
        logging.debug("Matching csv type to config")
        for column_conf_key in column_configs:
            columns = column_configs[column_conf_key]["columns"]
            columns_string = ",".join(columns)
            index_columns_string = "index" + columns_string
            if columns_line in (columns_string, index_columns_string):
                columns_list = columns
                data = column_configs[column_conf_key]["datafield"]
                logging.debug("Match found")
                break

        logging.debug("Reading to datafile")
        df = pd.read_csv(csv_document, header=0, names=columns_list)
        # logging.debug(df.head())
        # logging.debug(df[data].head())

        for parse_filter in parse_filters:
            filter_iterator = iter(parse_filter)
            parse_regex = next(filter_iterator)
            parse_replacment = next(filter_iterator)
            logging.debug(f"Applying filter: {parse_regex}")
            df[data] = df[data].replace(
                to_replace=parse_filter[parse_regex], value=parse_filter[parse_replacment], regex=True
            )
        base = splitext(csv_document)[0]
        doc_path = base + ".txt"
        logging.debug("Writing to file")
        with open(file=doc_path, mode="a", encoding="utf-8") as doc_file:
            for line in df[data].to_numpy():
                clean_text = extract(line)
                if clean_text is not None:
                    doc_file.write(clean_text + "\n\n")
                # logging.info(clean_text)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Parse csv file to a text file and filter out noise from web scrapes.")

    # Add arguments
    parser.add_argument(
        "--data-directory",
        type=str,
        default="./run_files/documents/csv_test",
        help="The directory where your csv files are stored",
    )

    parser.add_argument(
        "--parse-config-directory",
        type=str,
        default="./run_files/parse_configs/",
        help="The parse config directory",
    )

    parser.add_argument(
        "--parse-config-file",
        type=str,
        default="csv_columns.json",
        help="The parse config file",
    )

    parser.add_argument(
        "--filter-config-directory",
        type=str,
        default="./run_files/filters/",
        help="The parse config directory",
    )

    parser.add_argument(
        "--filter-config-file",
        type=str,
        default="web_scrape_filter.json",
        help="The parse config file",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        parse_config_directory=args.parse_config_directory,
        parse_config_file=args.parse_config_file,
        filter_config_directory=args.filter_config_directory,
        filter_config_file=args.filter_config_file,
    )
