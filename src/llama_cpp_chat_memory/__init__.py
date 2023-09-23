from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from os import getenv, mkdir
from os.path import exists, join, splitext, realpath, dirname
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import json
import yaml
import fnmatch
import base64
import toml

load_dotenv(find_dotenv())
CARD_AVATAR = None
CHARACTER_NAME = getenv("CHARACTER_NAME")

txt_template = """
{prompt_content}
Current conversation:
{history}

Question: {input}

### Response:
"""

card_template = """
### Instruction:
Continue the chat dialogue below. Write {character}'s next reply in a chat between User and {character}. Write a single reply only.

### Input:
Description:
{description}

Scenario:
{scenario}

Message Examples:
{mes_example}

Current conversation:
{history}

Question: {input}

### Response:
"""

def parse_prompt():
    #Currently the chat welcome message is read from chainlit.md file.
    script_root_path = dirname(realpath(__file__))
    help_file_path = join(script_root_path, "chainlit.md")
    config_toml_path = join(script_root_path, ".chainlit", "config.toml")
    prompt_dir = getenv("CHARACTER_CARD_DIR")
    prompt_name = getenv("CHARACTER_CARD")
    prompt_source =join(prompt_dir, prompt_name) 
    extension = splitext(prompt_source)[1]
    match extension:
        case ".txt":
            #No initial messages for text prompts. Live with it
            with open(help_file_path, "w") as w:
                w.write("")
            with open(prompt_source) as f:
                prompt_file = f.read()
            text_prompt = PromptTemplate(template=txt_template, input_variables=["prompt_content", "history", "input"])
            with open(config_toml_path, "r") as toml_file:
                toml_dict = toml.load(toml_file)
                toml_dict["UI"]["name"] = getenv("CHARACTER_NAME")

            with open(config_toml_path, "w") as toml_file:
                toml.dump(toml_dict, toml_file)
            return text_prompt.partial(prompt_content=prompt_file,)
        case ".json":
            with open(prompt_source) as f:
                prompt_file = f.read()
            card = json.loads(prompt_file)
        case ".yaml":
            with open(prompt_source) as f:
                card=yaml.safe_load(f)
        case ".png":
            global CARD_AVATAR
            is_v2=False
            if fnmatch.fnmatch(prompt_source, "*spec_v2.png"):
                is_v2=True
            elif fnmatch.fnmatch(prompt_source, "*tavern.png"):
                is_v2=False
            else:
                print("ERROR")
                #TODO ERROR
            im = Image.open(prompt_source)
            im.load()
            card = None
            if im.info is not None and "chara" in im.info:
                decoded = base64.b64decode(im.info["chara"])
                card = json.loads(decoded)
                if is_v2 and 'data' in card:
                    card = card['data']
            char_name = card["name"] if "name" in card else card["char_name"]
            temp_folder_path = join(script_root_path, "temp")
            copy_image_filename = join(temp_folder_path, char_name + ".png")
            if not exists(copy_image_filename):
                if not exists(temp_folder_path):
                    mkdir(temp_folder_path)
                data = list(im.getdata())
                image2 = Image.new(im.mode, im.size)
                image2.putdata(data)
                #Save a card image without metadata into temp
                image2.save(copy_image_filename)
                #Set the card image without metadata as avatar image
                CARD_AVATAR = copy_image_filename
            else:
                CARD_AVATAR = copy_image_filename

    prompt = PromptTemplate(template=card_template, input_variables=["character", "description", "scenario", "mes_example", "history", "input"])
    global CHARACTER_NAME
    char_name = card["name"] if "name" in card else card["char_name"]
    CHARACTER_NAME = char_name

    with open(config_toml_path, "r") as toml_file:
        toml_dict = toml.load(toml_file)
        toml_dict["UI"]["name"] = char_name

    with open(config_toml_path, "w") as toml_file:
        toml.dump(toml_dict, toml_file)

    description = card["description"] if "description" in card else card["char_persona"]
    scenario = card["scenario"] if "scenario" in card else card["world_scenario"]
    mes_example = card["mes_example"] if "mes_example" in card else card["example_dialogue"]
    first_message = card["first_mes"] if "first_mes" in card else card["char_greeting"]
    description = description.replace("{{user}}","User")
    scenario = scenario.replace("{{user}}","User")
    mes_example = mes_example.replace("{{user}}","User")
    first_message = first_message.replace("{{user}}","User")

    description = description.replace("{{User}}","User")
    scenario = scenario.replace("{{User}}","User")
    mes_example = mes_example.replace("{{User}}","User")
    first_message = first_message.replace("{{User}}","User")

    description = description.replace("{{char}}",char_name)
    scenario = scenario.replace("{{char}}",char_name)
    mes_example = mes_example.replace("{{char}}",char_name)
    first_message = first_message.replace("{{char}}",char_name)

    description = description.replace("{{Char}}",char_name)
    scenario = scenario.replace("{{Char}}",char_name)
    mes_example = mes_example.replace("{{Char}}",char_name)
    first_message = first_message.replace("{{Char}}",char_name)

    with open(help_file_path, "w") as w:
        w.write(first_message)
    return prompt.partial(character = char_name,description = description, scenario = scenario, mes_example = mes_example) 

def get_avatar_image():
    if CARD_AVATAR is None:
        prompt_dir = getenv("CHARACTER_CARD_DIR")
        prompt_name = getenv("CHARACTER_CARD")
        prompt_source =join(prompt_dir, prompt_name)
        base = splitext(prompt_source)[0]
        if exists(base + ".png"):
            return base + ".png"
        elif exists(base + ".jpg"):
            return base + ".jpg"
        return ""
    else:
        return CARD_AVATAR

def instantiate_llm():
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source =join(model_dir, model) 
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
        **params,
    )
    return llm_model_init

PROMPT = parse_prompt()
AVATAR_IMAGE = get_avatar_image()
USE_AVATAR_IMAGE = exists(AVATAR_IMAGE)
LLM_MODEL = instantiate_llm()
