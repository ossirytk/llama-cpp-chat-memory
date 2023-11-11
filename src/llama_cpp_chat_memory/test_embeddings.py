import argparse
import base64
import fnmatch
import json
import logging
from os import getenv, mkdir
from os.path import dirname, exists, join, realpath, splitext

import chromadb
import toml
import yaml
from chromadb.config import Settings
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from PIL import Image

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())
CARD_AVATAR = None
CHARACTER_NAME = getenv("CHARACTER_NAME")

question_generation_template = """
{llama_instruction}
Continue the chat dialogue below. Write {character}'s next reply in a chat between User and {character}. Write a single reply only.

{llama_input}
Description:
{description}

Scenario:
{scenario}

Message Examples:
{mes_example}

Context:
{vector_context}

Current conversation:
{history}

Question: {input}

{llama_response}
"""


def parse_prompt():
    # Currently the chat welcome message is read from chainlit.md file.
    script_root_path = dirname(realpath(__file__))
    help_file_path = join(script_root_path, "chainlit.md")
    config_toml_path = join(script_root_path, ".chainlit", "config.toml")
    prompt_dir = getenv("CHARACTER_CARD_DIR")
    prompt_name = getenv("CHARACTER_CARD")
    prompt_source = join(prompt_dir, prompt_name)
    extension = splitext(prompt_source)[1]
    match extension:
        case ".json":
            with open(prompt_source) as f:
                prompt_file = f.read()
            card = json.loads(prompt_file)
        case ".yaml":
            with open(prompt_source) as f:
                card = yaml.safe_load(f)
        case ".png":
            global CARD_AVATAR
            is_v2 = False
            if fnmatch.fnmatch(prompt_source, "*v2.png"):
                is_v2 = True
            elif fnmatch.fnmatch(prompt_source, "*tavern.png"):
                is_v2 = False
            else:
                logging.error("ERROR")
                # TODO ERROR
            im = Image.open(prompt_source)
            im.load()
            card = None
            if im.info is not None and "chara" in im.info:
                decoded = base64.b64decode(im.info["chara"])
                card = json.loads(decoded)
                if is_v2 and "data" in card:
                    card = card["data"]
            char_name = card["name"] if "name" in card else card["char_name"]
            temp_folder_path = join(script_root_path, "temp")
            copy_image_filename = join(temp_folder_path, char_name + ".png")
            if not exists(copy_image_filename):
                if not exists(temp_folder_path):
                    mkdir(temp_folder_path)
                data = list(im.getdata())
                image2 = Image.new(im.mode, im.size)
                image2.putdata(data)
                # Save a card image without metadata into temp
                image2.save(copy_image_filename)
                # Set the card image without metadata as avatar image
                CARD_AVATAR = copy_image_filename
            else:
                CARD_AVATAR = copy_image_filename

    prompt = PromptTemplate(template=question_generation_template, input_variables=[
        "character",
        "description",
        "scenario",
        "mes_example",
        "input",
        "llama_input",
        "llama_instruction",
        "llama_response",
        "vector_context",
        "history"]
        )
    global CHARACTER_NAME
    char_name = card["name"] if "name" in card else card["char_name"]
    CHARACTER_NAME = char_name

    with open(config_toml_path) as toml_file:
        toml_dict = toml.load(toml_file)
        toml_dict["UI"]["name"] = char_name

    with open(config_toml_path, "w") as toml_file:
        toml.dump(toml_dict, toml_file)

    if getenv("MODEL_TYPE") == "llama":
        llama_instruction = "### Instruction:"
        llama_input = "### Input:"
        llama_response = "### Response:"
    else:
        llama_instruction = ""
        llama_input = ""
        llama_response = ""
    description = card["description"] if "description" in card else card["char_persona"]
    scenario = card["scenario"] if "scenario" in card else card["world_scenario"]
    mes_example = card["mes_example"] if "mes_example" in card else card["example_dialogue"]
    first_message = card["first_mes"] if "first_mes" in card else card["char_greeting"]
    description = description.replace("{{user}}", "User")
    scenario = scenario.replace("{{user}}", "User")
    mes_example = mes_example.replace("{{user}}", "User")
    first_message = first_message.replace("{{user}}", "User")

    description = description.replace("{{User}}", "User")
    scenario = scenario.replace("{{User}}", "User")
    mes_example = mes_example.replace("{{User}}", "User")
    first_message = first_message.replace("{{User}}", "User")

    description = description.replace("{{char}}", char_name)
    scenario = scenario.replace("{{char}}", char_name)
    mes_example = mes_example.replace("{{char}}", char_name)
    first_message = first_message.replace("{{char}}", char_name)

    description = description.replace("You", char_name)
    scenario = scenario.replace("You", char_name)
    mes_example = mes_example.replace("You", char_name)
    first_message = first_message.replace("You", char_name)

    description = description.replace("{{Char}}", char_name)
    scenario = scenario.replace("{{Char}}", char_name)
    mes_example = mes_example.replace("{{Char}}", char_name)
    first_message = first_message.replace("{{Char}}", char_name)

    with open(help_file_path, "w") as w:
        w.write(first_message)
    return prompt.partial(
        character=char_name,
        description=description,
        scenario=scenario,
        mes_example=mes_example,
        llama_input=llama_input,
        llama_instruction=llama_instruction,
        llama_response=llama_response,
        vector_context=" "
        )


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

    llama = LlamaCppEmbeddings(model_path=model_source, **params,)
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    db = Chroma(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=llama
        )
    logging.info(f"There are {db._collection.count()} documents in the collection")
    logging.info("--------------------------------------------------------------------")
    # query it
    docs = db.similarity_search_with_score(query=query, k=k)

    vector_context = ""
    for answer in docs:
        logging.info("--------------------------------------------------------------------")
        logging.info(f"distance: {answer[1]}")
        logging.info(answer[0].metadata)
        logging.info(answer[0].page_content)
        vector_context = vector_context + answer[0].page_content
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
