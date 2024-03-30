import base64
import fnmatch
import json
import logging
from os import getenv, makedirs, mkdir
from os.path import dirname, exists, join, realpath, splitext

import chromadb
import pandas as pd
import toml
import yaml
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import load_prompt
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from PIL import Image

# Write logs to chat.log file. This is easier to read
# Requires that we empty the chainlit log handlers first
# If you want chainlit debug logs, you might need to disable this
LOGGIN_HANDLE = "chat"
if not exists("./logs/chat.log"):
    makedirs("./logs/", exist_ok=True)
    open("./logs/chat.log", "a").close()

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="./logs/chat.log",
    format="%(asctime)s %(levelname)s:%(message)s",
    encoding="utf-8",
    level=logging.INFO,
)
CHAT_LOG = logging.getLogger(LOGGIN_HANDLE)


load_dotenv(find_dotenv())
CARD_AVATAR = None
CHARACTER_NAME = None


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
            CHAT_LOG.debug(f"Loading filter list from: {key_storage_path}")
            CHAT_LOG.debug(f"Filter keys: {all_keys}")

    if all_keys is not None and "Content" in all_keys:
        return pd.DataFrame.from_dict(all_keys["Content"], orient="index", columns=["keys"])
    elif all_keys is not None and "Content" not in all_keys:
        return pd.DataFrame.from_dict(all_keys, orient="index", columns=["keys"])
    else:
        return None


def parse_question_refining_metadata_prompt():
    prompt_template_dir = getenv("PROMPT_TEMPLATE_DIRECTORY")
    prompt_metadata_template_name = "question_refining_metadata_template.json"
    metadata_prompt_template_path = join(prompt_template_dir, prompt_metadata_template_name)
    metadata_prompt = load_prompt(metadata_prompt_template_path)
    if getenv("MODEL_TYPE") == "alpaca":
        llama_instruction = "### Instruction:"
        llama_input = "### Input:"
        llama_response = "### Response:"
        llama_endtoken = ""
    elif getenv("MODEL_TYPE") == "mistral":
        llama_instruction = "[INST]\n"
        llama_input = ""
        llama_response = "[/INST]\n"
        llama_endtoken = ""
    else:
        llama_instruction = ""
        llama_input = ""
        llama_response = ""
        llama_endtoken = ""

    filled_metadata_prompt = metadata_prompt.partial(
        llama_input=llama_input,
        llama_instruction=llama_instruction,
        llama_response=llama_response,
        llama_endtoken=llama_endtoken,
        vector_context=" ",
    )
    return filled_metadata_prompt


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
    prompt_template_dir = getenv("PROMPT_TEMPLATE_DIRECTORY")
    prompt_template_name = getenv("PROMPT_TEMPLATE")
    prompt_template_path = join(prompt_template_dir, prompt_template_name)
    replace_you = getenv("REPLACE_YOU")
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
                error_message = f"Unrecognized card type for : {prompt_source}"
                CHAT_LOG.error("Could not load card info")
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

    prompt = load_prompt(prompt_template_path)

    global CHARACTER_NAME
    char_name = card["name"] if "name" in card else card["char_name"]
    CHARACTER_NAME = char_name

    with open(config_toml_path, encoding="utf-8") as toml_file:
        toml_dict = toml.load(toml_file)
        toml_dict["UI"]["name"] = char_name

    with open(file=config_toml_path, mode="w", encoding="utf-8") as toml_file:
        toml.dump(toml_dict, toml_file)

    # Supported model types are mistral and alpaca
    # Feel free to add things here
    if getenv("MODEL_TYPE") == "alpaca":
        llama_instruction = "### Instruction:"
        llama_input = "### Input:"
        llama_response = "### Response:"
        llama_endtoken = ""
    elif getenv("MODEL_TYPE") == "mistral":
        llama_instruction = "[INST]"
        llama_input = ""
        llama_response = "[/INST]\n"
        llama_endtoken = ""
    elif getenv("MODEL_TYPE") == "chatml":
        llama_instruction = "<|system|>"
        llama_input = "<|user|>"
        llama_response = "<|assistant|>\n"
        llama_endtoken = "</s>"
    else:
        llama_instruction = ""
        llama_input = ""
        llama_response = ""
        llama_endtoken = ""
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

    description = description.replace("{{Char}}", char_name)
    scenario = scenario.replace("{{Char}}", char_name)
    mes_example = mes_example.replace("{{Char}}", char_name)
    first_message = first_message.replace("{{Char}}", char_name)

    # Some poorly written cards just use You for character
    if replace_you:
        description = description.replace("You", "User")
        scenario = scenario.replace("You", "User")
        mes_example = mes_example.replace("You", "User")
        first_message = first_message.replace("You", "User")

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
        llama_endtoken=llama_endtoken,
        vector_context=" ",
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
    embeddings_model = getenv("EMBEDDINGS_MODEL")

    client = chromadb.PersistentClient(path=getenv("PERSIST_DIRECTORY"), settings=Settings(anonymized_telemetry=False))

    embeddings_type = getenv("EMBEDDINGS_TYPE")

    if embeddings_type == "llama":
        CHAT_LOG.info("Using llama embeddigs")
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
        CHAT_LOG.info("Using spacy embeddigs")
        # embedder = CustomSpacyEmbeddings(model_path="en_core_web_lg")
        embedder = CustomSpacyEmbeddings(model_path=embeddings_model)
    elif embeddings_type == "huggingface":
        CHAT_LOG.info("Using huggingface embeddigs")
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embedder = HuggingFaceEmbeddings(
            model_name=embeddings_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    else:
        error_message = f"Unsupported embeddings type: {embeddings_type}"
        raise ValueError(error_message)
    db = Chroma(
        client=client,
        collection_name=getenv("COLLECTION"),
        persist_directory=getenv("PERSIST_DIRECTORY"),
        embedding_function=embedder,
    )

    return db


def instantiate_llm():
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)

    # Add things here if you want to play with the model params
    # MAX_TOKENS is an optional param for when model answer cuts off
    # This can happen when large context models are told to print multiple paragraphs
    # Setting MAX_TOKENS lower than the context size can sometimes fix this
    params = {
        "n_ctx": getenv("N_CTX"),
        "temperature": 0.6,
        "last_n_tokens_size": 256,
        "n_batch": 1024,
        "repeat_penalty": 1.17647,
        "n_gpu_layers": getenv("LAYERS"),
        "rope_freq_scale": getenv("ROPE_CONTEXT"),
    }

    if getenv("USE_MAX_TOKENS"):
        CHAT_LOG.debug("Using max tokens")
        params["max_tokens"] = getenv("MAX_TOKENS")
    else:
        CHAT_LOG.debug("Not using max tokens")

    llm_model_init = LlamaCpp(
        model_path=model_source,
        streaming=True,
        **params,
    )
    return llm_model_init


ALL_KEYS = parse_keys()
QUSTION_REFINING_METADATA_PROMPT = parse_question_refining_metadata_prompt()
PROMPT = parse_prompt()
AVATAR_IMAGE = get_avatar_image()
USE_AVATAR_IMAGE = exists(AVATAR_IMAGE)
RETRIEVER = instantiate_retriever()
LLM_MODEL = instantiate_llm()
