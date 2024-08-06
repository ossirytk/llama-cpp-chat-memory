import base64
import fnmatch
import json
import logging
import re
from collections import deque
from functools import partial
from os import getcwd, getenv, makedirs, mkdir
from os.path import dirname, exists, join, realpath, splitext

import chainlit as cl
import chromadb
import pandas as pd
import spacy
import yaml
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from document_parsing.extract import entities, ngrams, terms
from document_parsing.extract.basics import terms_to_strings
from dotenv import find_dotenv, load_dotenv
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, load_prompt
from nltk import regexp_tokenize
from PIL import Image
from rank_bm25 import BM25Okapi


class ConveresationManager:
    def __init__(self, **kwargs):
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

        if "test" not in kwargs:
            # Character card details
            self.character_name: str = ""
            self.description: str = ""
            self.scenario: str = ""
            self.mes_example: str = ""
            self.llama_input: str = ""
            self.llama_instruction: str = ""
            self.llama_response: str = ""
            self.llama_endtoken: str = ""

            # Collections and template
            self.prompt_template_dir: str = ""
            self.prompt_template: str = ""

            # Init things
            self.collections_config = self.parse_collections_config()
            self.mes_collection_name = self.collections_config["mex_default"]
            self.context_collection_name = self.collections_config["context_default"]
            if self.mes_collection_name != "none" or self.context_collection_name != "none":
                self.embedder = self.instantiate_embeddings()
                self.use_embeddings = True
            else:
                self.use_embeddings = False
            if self.collections_config["mex_default"] != "none":
                self.all_mes_keys = self.parse_keys("mex_default")
                self.mes_retriever = self.instantiate_retriever("mex_default")
                self.use_mes = True
            else:
                self.use_mes = False
            if self.collections_config["context_default"] != "none":
                self.all_context_keys = self.parse_keys("context_default")
                self.context_retriever = self.instantiate_retriever("context_default")
                self.use_context = True
            else:
                self.use_context = False
            self.question_refining_prompt = self.parse_question_refining_metadata_prompt()
            self.prompt = self.parse_prompt()
            self.llm_model = self.instantiate_llm()
            self.historylen = int(getenv("BUFFER_K"))
            self.vector_sort_type = getenv("VECTOR_SORT_TYPE")
            self.user_message_history: deque[str] = deque(maxlen=self.historylen)
            self.ai_message_history: deque[str] = deque(maxlen=self.historylen)
            output_parser = StrOutputParser()

            # Initial chains
            self.conversation_chain = self.prompt | self.llm_model
            self.conversation_chain_test = self.prompt | self.llm_model | output_parser
            self.refine_type = getenv("REFINE_TYPE")
            self.refine_model = spacy.load(getenv("REFINE_MODEL"))
            self.parse_spacy_refining_config()
        else:
            # Character card details
            self.character_name: str = ""
            self.description: str = ""
            self.scenario: str = ""
            self.mes_example: str = ""
            self.llama_input: str = ""
            self.llama_instruction: str = ""
            self.llama_response: str = ""
            self.llama_endtoken: str = ""

            # Collections and template
            self.prompt_template_dir: str = ""
            self.prompt_template: str = ""

            # Init things
            self.collections_config = self.parse_collections_config()
            self.mes_collection_name = self.collections_config["mex_default"]
            self.context_collection_name = self.collections_config["context_default"]
            if self.mes_collection_name != "none" or self.context_collection_name != "none":
                self.embedder = self.instantiate_embeddings()
                self.use_embeddings = True
            else:
                self.use_embeddings = False
            if self.collections_config["mex_default"] != "none":
                self.all_mes_keys = self.parse_keys("mex_default")
                self.mes_retriever = self.instantiate_retriever("mex_default")
                self.use_mes = True
            else:
                self.use_mes = False
            if self.collections_config["context_default"] != "none":
                self.all_context_keys = self.parse_keys("context_default")
                self.context_retriever = self.instantiate_retriever("context_default")
                self.use_context = True
            else:
                self.use_context = False
            self.prompt = self.parse_prompt()
            self.mes_retriever = self.instantiate_retriever("mex_default")
            self.context_retriever = self.instantiate_retriever("context_default")
            self.vector_sort_type = getenv("VECTOR_SORT_TYPE")

    def parse_spacy_refining_config(self):
        parse_config_path = join(".", getenv("REFINE_CONFIG"))
        if exists(parse_config_path):
            with open(parse_config_path) as key_file:
                filter_content = key_file.read()
            filter_configs = json.loads(filter_content)
        else:
            logging.info("Could not load parse config file")
            return

        self.ngrams_list = filter_configs["ngs"]
        self.entities_list = filter_configs["entities"]
        self.noun_chunks = filter_configs["noun_chunks"]
        self.extract_type = filter_configs["extract_type"]

    def parse_collections_config(self):
        collection_config_path = getenv("COLLECTION_CONFIG")
        config_json = None
        if exists(collection_config_path):
            with open(collection_config_path) as key_file:
                content = key_file.read()
            config_json = json.loads(content)
        return config_json

    # Get metadata filter keys for this collection
    def parse_keys(self, vector_type: str) -> pd.DataFrame:
        if vector_type in ("mex_default", "context_default"):
            collection_name = self.collections_config[vector_type]
            self.chat_log.debug("Using default keys")
        else:
            collection_name = vector_type
            self.chat_log.debug(f"Using keys for: {vector_type}")
        all_keys = None
        if collection_name != "none":
            key_storage = getenv("KEY_STORAGE_DIRECTORY")
            makedirs(key_storage, exist_ok=True)
            key_storage_path = join(key_storage, collection_name + ".json")
            if exists(key_storage_path):
                with open(key_storage_path) as key_file:
                    content = key_file.read()
                all_keys = json.loads(content)
                self.chat_log.info(f"Loading filter list from: {key_storage_path}")
                self.chat_log.debug(f"Filter keys: {all_keys}")

        if all_keys is not None and "Content" in all_keys:
            return pd.DataFrame.from_dict(all_keys["Content"], orient="index", columns=["keys"])
        elif all_keys is not None and "Content" not in all_keys:
            return pd.DataFrame.from_dict(all_keys, orient="index", columns=["keys"])
        else:
            error_message = "Could not load keys"
            raise ValueError(error_message)

    def parse_question_refining_metadata_prompt(self) -> BasePromptTemplate:
        # TODO RUN CONFIG PROMPT TEMPLATE
        self.prompt_template_dir = getenv("PROMPT_TEMPLATE_DIRECTORY")
        makedirs(self.prompt_template_dir, exist_ok=True)
        prompt_metadata_template_name = "question_refining_metadata_template.json"
        metadata_prompt_template_path = join(self.prompt_template_dir, prompt_metadata_template_name)
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

    def parse_prompt(self) -> BasePromptTemplate:
        # Currently the chat welcome message is read from chainlit.md file.
        self.prompt_template_dir = getenv("PROMPT_TEMPLATE_DIRECTORY")
        makedirs(self.prompt_template_dir, exist_ok=True)
        script_root_path = dirname(realpath(__file__))
        help_file_path = join(script_root_path, "chainlit.md")
        prompt_dir = getenv("CHARACTER_CARD_DIR")
        makedirs(prompt_dir, exist_ok=True)
        prompt_name = getenv("CHARACTER_CARD")
        prompt_source = join(prompt_dir, prompt_name)

        self.prompt_template = self.collections_config["prompt_template_default"]
        prompt_template_path = join(self.prompt_template_dir, self.prompt_template)
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
                char_name: str = card["name"] if "name" in card else card["char_name"]
                cwd = getcwd()
                temp_folder_path = join(cwd, "public", "avatars")
                makedirs(temp_folder_path, exist_ok=True)
                clean_name = char_name.lower().replace(" ", "_")
                copy_image_filename = join(temp_folder_path, clean_name + ".png")
                if not exists(copy_image_filename):
                    if not exists(temp_folder_path):
                        mkdir(temp_folder_path)
                    data = list(im.getdata())
                    image2 = Image.new(im.mode, im.size)
                    image2.putdata(data)
                    # Save a card image without metadata into temp
                    image2.save(copy_image_filename)

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
            llama_response = "[/INST]"
            llama_endtoken = ""
        elif getenv("MODEL_TYPE") == "chatml":
            llama_instruction = "<|system|>"
            llama_input = "<|user|>"
            llama_response = "<|assistant|>"
            llama_endtoken = "</s>"
        elif getenv("MODEL_TYPE") == "solar":
            llama_instruction = ""
            llama_input = "<s> ### User:"
            llama_response = "### Assistant:"
            llama_endtoken = ""
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

        self.description = description
        self.scenario = scenario
        self.mes_example = mes_example
        self.llama_input = llama_input
        self.llama_instruction = llama_instruction
        self.llama_response = llama_response
        self.llama_endtoken = llama_endtoken

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

    def change_prompt(self, new_template: str):
        prompt_template_path = join(self.prompt_template_dir, new_template)
        prompt = load_prompt(prompt_template_path)
        self.prompt = prompt.partial(
            character=self.character_name,
            description=self.description,
            scenario=self.scenario,
            mes_example=self.mes_example,
            llama_input=self.llama_input,
            llama_instruction=self.llama_instruction,
            llama_response=self.llama_response,
            llama_endtoken=self.llama_endtoken,
            vector_context=" ",
        )
        self.prompt_template = new_template

    def instantiate_embeddings(self) -> Embeddings:
        model_dir = getenv("MODEL_DIR")
        makedirs(model_dir, exist_ok=True)
        model = getenv("MODEL")
        model_source = join(model_dir, model)
        embeddings_model = getenv("EMBEDDINGS_MODEL")

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

        return embedder

    def instantiate_retriever(self, retriever_string: str) -> Chroma:
        if retriever_string in ("mex_default", "context_default"):
            self.chat_log.info("instantiate retriver with default")
            collection_name = self.collections_config[retriever_string]
        else:
            self.chat_log.info(f"Setting retriever to: {retriever_string}")
            collection_name = retriever_string

        if not self.use_embeddings or collection_name == "none":
            self.chat_log.info("Embedder is None or collection name is none")
            return None

        persist_directory = getenv("PERSIST_DIRECTORY")
        makedirs(persist_directory, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

        db = Chroma(
            client=client,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedder,
        )

        return db

    def instantiate_llm(self) -> LlamaCpp:
        model_dir = getenv("MODEL_DIR")
        makedirs(model_dir, exist_ok=True)
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

    def calculate_fusion_rank(self, query: str, docs: list[Document]) -> pd.DataFrame:
        text_list = []
        vector_sort_type = self.vector_sort_type

        for document, _distance in docs:
            text_list.append(document.page_content)

        tokenized_corpus = [doc.split(" ") for doc in text_list]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        sorted_doc_scores = sorted(doc_scores, reverse=True)

        fusion_scores = []
        vector_distance_scores = []
        bm25_scores = []
        for idx, bm25_value in enumerate(doc_scores):
            vector_distance_rank = idx + 1
            bm_25_rank = sorted_doc_scores.index(bm25_value) + 1
            fusion_rank = (1.0 / (0.6 + vector_distance_rank)) + (1.0 / (0.6 + bm_25_rank))
            vector_distance_scores.append(vector_distance_rank)
            bm25_scores.append(bm_25_rank)
            fusion_scores.append(fusion_rank)

        df = pd.DataFrame(
            {
                "content": text_list,
                "vector_distance_rank": vector_distance_scores,
                "bm25_rank": bm25_scores,
                "fusion_rank": fusion_scores,
            }
        )
        # Default order is vector distance
        if vector_sort_type in ["bm25_rank", "fusion_rank"]:
            df = df.sort_values(vector_sort_type, ascending=False)
        return df

    def get_metadata_filter(self, metadata_result: str, filter_type: str) -> str:
        if filter_type == "mes":
            all_keys = self.all_mes_keys
        else:
            all_keys = self.all_context_keys

        self.chat_log.info(f"Query metadata {metadata_result}")
        # Currently Chroma has no "like" implementation so this is a case sensitive hack with uuids
        # There is also an issue when filter has only one item since "in" expects multiple items
        # With one item, just use a dict with "uuid", "filter"
        filter_dict = {}
        metadata_result = re.sub("Keywords?:?|keywords?:?|\\[.*\\]", "", metadata_result)

        key_split = metadata_result.split()
        if len(key_split) > 1:
            tokens = regexp_tokenize(metadata_result, r"\w+", gaps=False)
            keys_df = all_keys[all_keys["keys"].isin(tokens)]
            keys_dict = keys_df.to_dict()
            filter_dict = keys_dict["keys"]
        elif len(key_split) == 1:
            keys_df = all_keys[all_keys["keys"] == metadata_result]
            keys_dict = keys_df.to_dict()
            filter_dict = keys_dict["keys"]

        where = ""

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
        return where

    def get_vector(
        self, message: str, filter_type: str, metadata_filter: str, k_size: int
    ) -> list[tuple[Document, float]]:
        if filter_type == "mes":
            retriever = self.mes_retriever
        else:
            retriever = self.context_retriever

        self.chat_log.info(f"There are {retriever._collection.count()} documents in the collection")
        self.chat_log.info(f"Filter {metadata_filter}")

        k_buffer = k_size + 4
        docs = retriever.similarity_search_with_score(query=message, k=k_buffer, filter=metadata_filter)
        return docs

    def get_history(self) -> str:
        message_history = ""
        user_message_history_list = list(self.user_message_history)
        ai_message_history_list = list(self.ai_message_history)
        for x in range(len(user_message_history_list)):
            user_message = user_message_history_list[x]
            ai_message = ai_message_history_list[x]
            new_line = "User: " + user_message + "\n" + self.character_name + ":" + ai_message + "\n"
            message_history = message_history + new_line
        return message_history

    def update_history(self, message, result):
        self.user_message_history.append(message)
        self.ai_message_history.append(result)

    async def ask_question_test(self, message: str):
        self.chat_log.info(message)
        vector_k = int(getenv("VECTOR_K"))

        self.chat_log.info(f"Query {message}")
        query_message = message.replace(self.character_name, "")

        doc = self.refine_model(query_message)
        extracted_terms = terms(
            doc,
            ngs=partial(ngrams, n=self.noun_chunks, include_pos=self.ngrams_list),
            ents=partial(
                entities,
                include_types=self.entities_list,
            ),
            dedupe=True,
        )
        metadata_terms = list(terms_to_strings(extracted_terms, by=self.extract_type))
        if len(metadata_terms) > 1:
            metadata_result = ", ".join(metadata_terms)
        elif len(metadata_terms) == 1:
            metadata_result = metadata_terms[0]
        else:
            metadata_result = ""

        mes_context = ""
        if self.use_mes:
            mes_filter = self.get_metadata_filter(metadata_result, "mes")
            mes_docs = self.get_vector(query_message, "mes", mes_filter, vector_k)
            if mes_docs is not None and len(mes_docs) > 1:
                mes_df: pd.DataFrame = self.calculate_fusion_rank(query_message, mes_docs)
                mes_df = mes_df.iloc[0:vector_k]
                mes_context = "\n".join(mes_df["content"].tolist())
            elif mes_docs is not None and len(mes_docs) == 1:
                mes_context = mes_docs[0][0].page_content

        vector_context = ""
        if self.use_context:
            context_filter = self.get_metadata_filter(metadata_result, "context")
            vector_docs = self.get_vector(query_message, "context", context_filter, vector_k)

            if vector_docs is not None and len(vector_docs) > 1:
                vector_context_df: pd.DataFrame = self.calculate_fusion_rank(query_message, vector_docs)
                vector_context_df = vector_context_df.iloc[0:vector_k]
                vector_context = "\n".join(vector_context_df["content"].tolist())
            elif vector_docs is not None and len(vector_docs) == 1:
                vector_context = vector_docs[0].page_content

        vector_context = vector_context.replace("{{char}}", self.character_name)
        mes_context = mes_context.replace("{{char}}", self.character_name)

        history = self.get_history()
        query_input = {
            "input": message,
            "history": history,
            "mes_example": mes_context,
            "vector_context": vector_context,
        }
        self.chat_log.info(f"query input: {query_input}")

        self.conversation_chain_test = self.prompt | self.llm_model
        chunks = []
        async for chunk in self.conversation_chain_test.astream(
            query_input,
        ):
            chunks.append(chunk)
            print(chunk, flush=True, end="")

    async def ask_question(self, message: cl.Message) -> cl.Message:
        self.chat_log.info(message.content)
        vector_k = int(getenv("VECTOR_K"))

        query_message = message.content.replace(self.character_name, "")

        doc = self.refine_model(query_message)
        extracted_terms = terms(
            doc,
            ngs=partial(ngrams, n=self.noun_chunks, include_pos=self.ngrams_list),
            ents=partial(
                entities,
                include_types=self.entities_list,
            ),
            dedupe=True,
        )
        metadata_terms = list(terms_to_strings(extracted_terms, by=self.extract_type))
        metadata_result = ", ".join(metadata_terms)

        mes_context = ""
        if self.use_mes:
            mes_filter = self.get_metadata_filter(metadata_result, "mes")
            mes_docs = self.get_vector(query_message, "mes", mes_filter, vector_k)
            if mes_docs is not None and len(mes_docs) > 1:
                mes_df: pd.DataFrame = self.calculate_fusion_rank(query_message, mes_docs)
                mes_df = mes_df.iloc[0:vector_k]
                mes_context = "\n".join(mes_df["content"].tolist())
            elif mes_docs is not None and len(mes_docs) == 1:
                mes_context = mes_docs[0][0].page_content

        vector_context = ""
        if self.use_context:
            context_filter = self.get_metadata_filter(metadata_result, "context")
            vector_docs = self.get_vector(query_message, "context", context_filter, vector_k)
            if vector_docs is not None and len(vector_docs) > 1:
                vector_context_df: pd.DataFrame = self.calculate_fusion_rank(query_message, vector_docs)
                vector_context_df = vector_context_df.iloc[0:vector_k]
                vector_context = "\n".join(vector_context_df["content"].tolist())
            elif vector_docs is not None and len(vector_docs) == 1:
                vector_context = vector_docs[0].page_content

        vector_context = vector_context.replace("{{char}}", self.character_name)
        mes_context = mes_context.replace("{{char}}", self.character_name)

        history = self.get_history()
        query_input = {
            "input": message.content,
            "history": history,
            "mes_example": mes_context,
            "vector_context": vector_context,
        }
        self.chat_log.info(f"query input: {query_input}")

        self.conversation_chain_test = self.prompt | self.llm_model
        result = cl.Message(content="")

        async for chunk in self.conversation_chain_test.astream(
            query_input,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await result.stream_token(chunk)
        self.update_history(message.content, result.content)
        return result

    def get_character_name(self) -> str:
        return self.character_name

    def update_settings(self, settings: dict[str, str]):
        self.chat_log.info("Updating settings")
        self.chat_log.debug(
            f"Current Settings: {self.prompt_template}, {self.mes_collection_name} and {self.context_collection_name}"
        )

        # TODO does this actually correctly disable with none?

        if settings["prompt_template_options"] != self.prompt_template:
            self.change_prompt(settings["prompt_template_options"])
            self.chat_log.info(f"Prompte template changed to: {settings['prompt_template_options']}")

        if settings["mex_collection"] == "none":
            self.use_mes = False
        elif settings["mex_collection"] != self.mes_collection_name:
            self.all_mes_keys = self.parse_keys(settings["mex_collection"])
            self.mes_retriever = self.instantiate_retriever(settings["mex_collection"])
            self.use_mes = True
            self.chat_log.info(f"Mex collection changed to: {settings['mex_collection']}")

        if settings["context_collection"] == "none":
            self.use_context = False
        elif settings["context_collection"] != self.context_collection_name:
            self.all_context_keys = self.parse_keys(settings["context_collection"])
            self.context_retriever = self.instantiate_retriever(settings["context_collection"])
            self.use_context = True
            self.chat_log.info(f"Context collection changed to: {settings['context_collection']}")

    def get_prompt_templates(self) -> list[str]:
        return self.collections_config["prompt_template_options"]

    def get_prompt_template_index(self) -> int:
        return self.collections_config["prompt_template_options"].index(
            self.collections_config["prompt_template_default"]
        )

    def get_context_collections(self) -> list[str]:
        return self.collections_config["context_options"]

    def get_context_index(self) -> int:
        return self.collections_config["context_options"].index(self.collections_config["context_default"])

    def get_mes_collections(self) -> list[str]:
        return self.collections_config["mex_options"]

    def get_mes_index(self) -> int:
        return self.collections_config["mex_options"].index(self.collections_config["mex_default"])
