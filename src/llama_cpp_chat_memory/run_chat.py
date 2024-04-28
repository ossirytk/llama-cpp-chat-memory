import base64
import fnmatch
import json
from os import getenv
from os.path import dirname, join, realpath, splitext

import toml
import yaml
from chainlit.cli import run_chainlit
from PIL import Image


def update_toml():
    script_root_path = dirname(realpath(__file__))
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
            is_v2 = False
            if fnmatch.fnmatch(prompt_source, "*v2.png"):
                is_v2 = True
            elif fnmatch.fnmatch(prompt_source, "*tavern.png"):
                is_v2 = False
            else:
                error_message = f"Unrecognized card type for : {prompt_source}"
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

    char_name = card["name"] if "name" in card else card["char_name"]

    with open(config_toml_path, encoding="utf-8") as toml_file:
        toml_dict = toml.load(toml_file)
        toml_dict["UI"]["name"] = char_name

    with open(file=config_toml_path, mode="w", encoding="utf-8") as toml_file:
        toml.dump(toml_dict, toml_file)


# Update toml with the character card name before running the chat application
update_toml()
# Chainlit loads the toml config before running the target,
# so updates to configs must be done before running

# TODO: There seems to be some cahching leftover in chainlit.
# To have character change take effect requires that you run run_chat twice
run_chainlit("character_chat.py")
