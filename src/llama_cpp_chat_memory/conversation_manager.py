import base64
import fnmatch
import json
import logging
import re
from os import getenv, makedirs, mkdir
from os.path import dirname, exists, join, realpath, splitext

import chromadb
import pandas as pd
import yaml
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import load_prompt
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from nltk import regexp_tokenize
from PIL import Image

# TODO: Move config file rewrites to test.py
# TODO: getters


class ConveresationManager:
    def __init__(self):
        # Write logs to chat.log file. This is easier to read
        # Requires that we empty the chainlit log handlers first
        # If you want chainlit debug logs, you might need to disable this
        logging_handle = "chat"
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
        self.chat_log = logging.getLogger(logging_handle)

        load_dotenv(find_dotenv())
        self.card_avatar = None
        self.character_name = None

        self.all_keys = self.parse_keys()
        self.question_refining_prompt = self.parse_question_refining_metadata_prompt()
        self.prompt = self.parse_prompt()
        self.avatar_image = self.get_avatar_image()
        self.use_avatar_image = exists(self.avatar_image)
        self.retriever = self.instantiate_retriever()
        self.llm_model = self.instantiate_llm()
        # Use basic conversation chain with buffered conversation fistory
        self.conversation_chain = ConversationChain(
            prompt=self.prompt,
            llm=self.llm_model,
            verbose=True,
            memory=ConversationBufferWindowMemory(
                k=int(getenv("BUFFER_K")), human_prefix="User", ai_prefix=self.character_name
            ),
        )

    # Get metadata filter keys for this collection
    def parse_keys(self):
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
                self.chat_log.debug(f"Loading filter list from: {key_storage_path}")
                self.chat_log.debug(f"Filter keys: {all_keys}")

        if all_keys is not None and "Content" in all_keys:
            return pd.DataFrame.from_dict(all_keys["Content"], orient="index", columns=["keys"])
        elif all_keys is not None and "Content" not in all_keys:
            return pd.DataFrame.from_dict(all_keys, orient="index", columns=["keys"])
        else:
            return None

    def parse_question_refining_metadata_prompt(self):
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

    def parse_prompt(self):
        # Currently the chat welcome message is read from chainlit.md file.
        script_root_path = dirname(realpath(__file__))
        help_file_path = join(script_root_path, "chainlit.md")
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
                is_v2 = False
                if fnmatch.fnmatch(prompt_source, "*v2.png"):
                    is_v2 = True
                elif fnmatch.fnmatch(prompt_source, "*tavern.png"):
                    is_v2 = False
                else:
                    error_message = f"Unrecognized card type for : {prompt_source}"
                    self.chat_log.error("Could not load card info")
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
                temp_folder_path = join(script_root_path, "run_files", "temp")
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
                    self.card_avatar = copy_image_filename
                else:
                    self.card_avatar = copy_image_filename

        prompt = load_prompt(prompt_template_path)

        char_name = card["name"] if "name" in card else card["char_name"]
        self.character_name = char_name

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

    def get_avatar_image(self):
        if self.card_avatar is None:
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
            return self.card_avatar

    def instantiate_retriever(self):
        if getenv("COLLECTION") == "":
            return None

        model_dir = getenv("MODEL_DIR")
        model = getenv("MODEL")
        model_source = join(model_dir, model)
        embeddings_model = getenv("EMBEDDINGS_MODEL")

        client = chromadb.PersistentClient(
            path=getenv("PERSIST_DIRECTORY"), settings=Settings(anonymized_telemetry=False)
        )

        embeddings_type = getenv("EMBEDDINGS_TYPE")

        if embeddings_type == "llama":
            self.chat_log.info("Using llama embeddigs")
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
            self.chat_log.info("Using spacy embeddigs")
            # embedder = CustomSpacyEmbeddings(model_path="en_core_web_lg")
            embedder = CustomSpacyEmbeddings(model_path=embeddings_model)
        elif embeddings_type == "huggingface":
            self.chat_log.info("Using huggingface embeddigs")
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

    def instantiate_llm(self):
        model_dir = getenv("MODEL_DIR")
        model = getenv("MODEL")
        model_source = join(model_dir, model)

        # Add things here if you want to play with the model params
        # MAX_TOKENS is an optional param for when model answer cuts off
        # This can happen when large context models are told to print multiple paragraphs
        # Setting MAX_TOKENS lower than the context size can sometimes fix this

        params = {
            "seed": getenv("SEED"),
            "n_ctx": getenv("N_CTX"),
            "last_n_tokens_size": getenv("LAST_N_TOKENS_SIZE"),
            "n_batch": getenv("N_BATCH"),
            "max_tokens": getenv("MAX_TOKENS"),
            "n_parts": getenv("N_PARTS"),
            "use_mlock": getenv("USE_MLOCK"),
            "use_mmap": getenv("USE_MMAP"),
            "top_p": getenv("TOP_P"),
            "top_k": getenv("TOP_K"),
            "temperature": getenv("TEMPERATURE"),
            "repeat_penalty": getenv("REPEAT_PENALTY"),
            "n_gpu_layers": getenv("LAYERS"),
            "rope_freq_scale": getenv("ROPE_CONTEXT"),
            "verbose": getenv("VERBOSE"),
        }

        llm_model_init = LlamaCpp(
            model_path=model_source,
            streaming=True,
            **params,
        )
        return llm_model_init

    def get_vector_context(self, message):
        vector_context = ""
        if self.retriever:
            # TODO rework this. The question refining prompt can have poor accuracy
            # Use ner?
            llm_chain_refine = LLMChain(prompt=self.question_refining_prompt, llm=self.llm_model)
            metadata_result = llm_chain_refine.invoke(message)
            metadata_query = metadata_result["text"]

            self.chat_log.info(f"Query {message}")
            self.chat_log.info(f"Query metadata {metadata_query}")
            # Currently Chroma has no "like" implementation so this is a case sensitive hack with uuids
            # There is also an issue when filter has only one item since "in" expects multiple items
            # With one item, just use a dict with "uuid", "filter"
            filter_dict = {}
            metadata_query = re.sub("Keywords?:?|keywords?:?|\\[.*\\]", "", metadata_query)
            if self.all_keys is not None and not self.all_keys.empty:
                tokens = regexp_tokenize(metadata_query.lower(), r"\w+", gaps=False)
                keys_df = self.all_keys[self.all_keys["keys"].isin(tokens)]
                keys_dict = keys_df.to_dict()
                filter_dict = keys_dict["keys"]

            if len(filter_dict) == 1:
                metadata_filter = {}
                for item in filter_dict.items():
                    metadata_filter[item[0]] = item[1]
                where = filter_dict
            elif len(filter_dict) > 1:
                metadata_filter_list = []
                for item in filter_dict.items():
                    metadata_filter_list.append({item[0]: {"$in": [item[1]]}})
                where = {"$or": metadata_filter_list}
            else:
                where = None

            k = int(getenv("VECTOR_K"))
            self.chat_log.info(f"There are {self.retriever._collection.count()} documents in the collection")
            self.chat_log.info(f"Filter {where}")

            docs = self.retriever.similarity_search_with_score(query=message, k=k, filter=where)
            for answer in docs:
                vector_context = vector_context + answer[0].page_content

            self.chat_log.info(vector_context)
            return vector_context

    def ask_question(self, message, callback):
        self.chat_log.info(message)
        vector_context = self.get_vector_context(message)
        self.conversation_chain.prompt = self.conversation_chain.prompt.partial(vector_context=vector_context)
        result = self.conversation_chain.invoke(message, callbacks=[callback])
        return result["response"]

    def get_character_name(self):
        return self.character_name

    def get_use_avatar_image(self):
        return self.use_avatar_image

    def get_avatar_image_path(self):
        return self.avatar_image