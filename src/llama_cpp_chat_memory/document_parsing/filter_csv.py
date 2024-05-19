import argparse
import glob
import json
import logging
import os
from os.path import exists, join

import pandas as pd
from dotenv import find_dotenv, load_dotenv

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

    for csv_document in documents_paths_csv:
        logging.debug(f"Processing: {csv_document}")
        columns_list = []
        with open(csv_document, encoding="utf8") as f:
            first_line = f.readline()
        columns_line = "index" + first_line.strip()
        logging.debug("Matching csv type to config")
        for column_conf_key in column_configs:
            columns = column_configs[column_conf_key]["columns"]
            columns_string = ",".join(columns)
            if columns_string == columns_line:
                columns_list = columns
                logging.debug("Match found")
                break

        logging.debug("Reading to datafile")
        df = pd.read_csv(csv_document, header=0, names=columns_list)
        item_count = df.shape[0]
        logging.debug(f"item count: {item_count}")
        logging.debug(df.head())

        for csv_filter in filter_configs["filters"]:
            if "whitelist" in csv_filter:
                whitelist = csv_filter["whitelist"]
                tags = csv_filter["filter_field"]
                df = df[df[tags].apply(lambda x, wordlist=set(whitelist): any(word in x for word in wordlist))]
                item_count = df.shape[0]
                logging.debug(f"item count: {item_count}")
                logging.debug(df.head())

            if "blacklist" in csv_filter:
                blacklist = csv_filter["blacklist"]
                tags = csv_filter["filter_field"]
                df = df[df[tags].apply(lambda x, wordlist=set(blacklist): not any(word in x for word in wordlist))]
                item_count = df.shape[0]
                logging.debug(f"item count: {item_count}")
                logging.debug(df.head())

        output = documents_directory + "/filtered.csv"
        df.to_csv(output, index=False)


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Filter rows from a csv file using whitelist and blacklist filters")

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
        default="csv_filter.json",
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
