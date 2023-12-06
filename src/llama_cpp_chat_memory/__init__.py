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
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from PIL import Image

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)
load_dotenv(find_dotenv())
CARD_AVATAR = None
CHARACTER_NAME = None

question_generation_template = """
{llama_instruction}
Continue the chat dialogue below. Write {character}'s next reply in a chat between User and {character}. Answer in first person. Write a single reply only.

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
"""  # noqa: E501

# Get metadata filter keys for this collection
def parse_keys():
    use_keys = getenv("USE_KEY_STORAGE")
    collection_name = getenv("COLLECTION")
    all_keys = None
    if use_keys:
        key_storage = getenv("KEY_STORAGE_DIRECTORY")
        key_storage_path = join(key_storage, collection_name + ".json")
        if exists(key_storage_path):
            with open(key_storage_path) as key_file:
                content = key_file.read()
            all_keys = json.loads(content)
            logging.debug(f"Loading filter list from: {key_storage_path}")
            logging.debug(f"Filter keys: {all_keys}")
    return all_keys


def parse_prompt():
    # Currently the chat welcome message is read from chainlit.md file.
    script_root_path = dirname(realpath(__file__))
    # TODO Improve this. Loading new cards seems to require reloading the card.
    # Might have to do with some racing condition where the md file gets loaded
    # before it gets updated
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
            # TODO a better implementation for the globals
            # Consider reloacating the content from init
            global CARD_AVATAR
            is_v2 = False
            if fnmatch.fnmatch(prompt_source, "*v2.png"):
                is_v2 = True
            elif fnmatch.fnmatch(prompt_source, "*tavern.png"):
                is_v2 = False
            else:
                error_message= f"Unrecognized card type for : {prompt_source}"
                logging.error("Could not load card info")
                raise ValueError(error_message)
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
        "history"
        ])
    global CHARACTER_NAME
    char_name = card["name"] if "name" in card else card["char_name"]
    CHARACTER_NAME = char_name

    with open(config_toml_path) as toml_file:
        toml_dict = toml.load(toml_file)
        toml_dict["UI"]["name"] = char_name

    with open(config_toml_path, "w") as toml_file:
        toml.dump(toml_dict, toml_file)

    # Supported model types are mistral and alpaca
    # Feel free to add things here
    if getenv("MODEL_TYPE") == "alpaca":
        llama_instruction = "### Instruction:"
        llama_input = "### Input:"
        llama_response = "### Response:"
    elif getenv("MODEL_TYPE") == "mistral":
        llama_instruction = "[INST]\n"
        llama_input = ""
        llama_response = "[/INST]\n"
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


def get_avatar_image():
    if CARD_AVATAR is None:
        prompt_dir = getenv("CHARACTER_CARD_DIR")
        prompt_name = getenv("CHARACTER_CARD")
        prompt_source = join(prompt_dir, prompt_name)
        base = splitext(prompt_source)[0]
        if exists(base + ".png"):
            return base + ".png"
        elif exists(base + ".jpg"):
            return base + ".jpg"
        return ""
    else:
        return CARD_AVATAR

def instantiate_retriever():
    if getenv("COLLECTION") == "":
        return None

    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)

    client = chromadb.PersistentClient(path=getenv("PERSIST_DIRECTORY"), settings=Settings(anonymized_telemetry=False))

    embeddings_type = getenv("EMBEDDINGS_TYPE")

    if embeddings_type == "llama":
        logging.info("Using llama embeddigs")
        params = {
            "n_ctx": getenv("N_CTX"),
            "n_batch": 1024,
            "n_gpu_layers": getenv("LAYERS"),
        }
        embedder = LlamaCppEmbeddings(
            model_path=model_source,
            **params,
        )
    elif embeddings_type == "spacy":
        logging.info("Using spacy embeddigs")
        embedder = CustomSpacyEmbeddings()
    else:
        error_message = f"Unsupported embeddings type: {embeddings_type}"
        raise ValueError(error_message)
    db = Chroma(
        client=client,
        collection_name=getenv("COLLECTION"),
        persist_directory=getenv("PERSIST_DIRECTORY"),
        embedding_function=embedder
        )

    return db


def instantiate_llm():
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)

    # Add things here if you want to play with the model params
    params = {
        "n_ctx": getenv("N_CTX"),
        "temperature": 0.6,
        "last_n_tokens_size": 256,
        "n_batch": 1024,
        "repeat_penalty": 1.17647,
        "n_gpu_layers": getenv("LAYERS"),
        "rope_freq_scale": getenv("ROPE_CONTEXT"),
    }

    llm_model_init = LlamaCpp(
        model_path=model_source,
        streaming=True,
        **params,
    )
    return llm_model_init

ALL_KEYS=parse_keys()
PROMPT = parse_prompt()
AVATAR_IMAGE = get_avatar_image()
USE_AVATAR_IMAGE = exists(AVATAR_IMAGE)
RETRIEVER = instantiate_retriever()
LLM_MODEL = instantiate_llm()
