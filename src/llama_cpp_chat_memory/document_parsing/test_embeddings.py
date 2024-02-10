import argparse
import json
import logging
from os import getenv
from os.path import exists, join

import chromadb
from chromadb.config import Settings
from custom_llm_classes.custom_spacy_embeddings import CustomSpacyEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain_community.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma

load_dotenv(find_dotenv())


# logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.DEBUG)
logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)


def get_refined_query_data(query: str, prompt_template_dir: str) -> tuple[str, str]:
    prompt_template_name = "question_refining_template2.json"
    prompt_template_path = join(prompt_template_dir, prompt_template_name)
    prompt_metadata_template_name = "question_refining_metadata_template.json"
    metadata_prompt_template_path = join(prompt_template_dir, prompt_metadata_template_name)

    prompt = load_prompt(prompt_template_path)
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

    filled_prompt = prompt.partial(
        llama_input=llama_input,
        llama_instruction=llama_instruction,
        llama_response=llama_response,
        vector_context=" ",
    )

    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)

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
        params["max_tokens"] = getenv("MAX_TOKENS")

    llm_model = LlamaCpp(
        model_path=model_source,
        streaming=True,
        **params,
    )

    llm_chain = LLMChain(prompt=filled_prompt, llm=llm_model)

    result = llm_chain.invoke(query)
    logging.info(f"Answer: {result['text']}")

    metadata_prompt = load_prompt(metadata_prompt_template_path)
    filled_metadata_prompt = metadata_prompt.partial(
        llama_input=llama_input,
        llama_instruction=llama_instruction,
        llama_response=llama_response,
        llama_endtoken=llama_endtoken,
        vector_context=" ",
    )

    llm_chain.prompt = filled_metadata_prompt
    metadata_result = llm_chain.invoke(result["text"])
    logging.info(f"Answer: {metadata_result['text']}")

    return (result["text"], metadata_result["text"])


def test_embeddings(
    refined_query: str,
    query_metadata: str,
    k: int,
    collection_name: str,
    persist_directory: str,
    embeddings_type: str,
    search_type,
) -> None:
    model_dir = getenv("MODEL_DIR")
    model = getenv("MODEL")
    model_source = join(model_dir, model)
    embeddings_model = getenv("EMBEDDINGS_MODEL")

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
        # embedder = CustomSpacyEmbeddings(model_path="en_core_web_lg")
        embedder = CustomSpacyEmbeddings(model_path=embeddings_model)
    elif embeddings_type == "huggingface":
        logging.info("Using huggingface embeddigs")
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embedder = HuggingFaceEmbeddings(
            model_name=embeddings_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    else:
        error_message = f"Unsupported embeddings type: {embeddings_type}"
        raise ValueError(error_message)
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    db = Chroma(
        client=client, collection_name=collection_name, persist_directory=persist_directory, embedding_function=embedder
    )

    use_keys = getenv("USE_KEY_STORAGE")
    all_keys = None
    if use_keys:
        key_storage = getenv("KEY_STORAGE_DIRECTORY")
        key_storage_path = join(".", key_storage, collection_name + ".json")
        if exists(key_storage_path):
            with open(key_storage_path, encoding="utf-8") as key_file:
                content = key_file.read()
            all_keys = json.loads(content)
        else:
            logging.info("Could not load filter list")
    logging.info(f"Loading filter list from: {key_storage_path}")
    # logging.debug(f"Filter keys: {all_keys}")

    # Currently Chroma has no "like" implementation so this is a case sensitive hack
    # There is also an issue with the filter only having one item so we use filter_list in this case
    metadata_filter_list = []
    filter_list = {}
    if all_keys:
        for item in all_keys.items():
            if item[1].lower() in query_metadata.lower():
                filter_list[item[0]] = item[1]
                metadata_filter_list.append({item[0]: {"$in": [item[1]]}})
    else:
        logging.info("No keys")

    if len(filter_list) == 1:
        where = filter_list
    elif len(filter_list) > 1:
        where = {"$or": metadata_filter_list}
    else:
        where = None
    # query it
    if search_type == "similiarity":
        logging.info("Similiarity search with score")
        logging.info(f"There are {db._collection.count()} documents in the collection")
        logging.debug(f"filter is: {where}")
        docs = db.similarity_search_with_score(query=refined_query, k=k, filter=where)
        vector_context = ""
        for answer in docs:
            logging.debug("--------------------------------------------------------------------")
            logging.debug(f"distance: {answer[1]}")
            logging.debug(answer[0].metadata)
            logging.debug(answer[0].page_content)
            vector_context = vector_context + answer[0].page_content
        logging.info("--------------------------------------------------------------------")
        logging.info(vector_context)
    elif search_type == "mmr":
        logging.info("Max marginal relevance search")
        logging.info(f"There are {db._collection.count()} documents in the collection")
        logging.debug(f"filter is: {where}")
        docs = db.max_marginal_relevance_search(query=refined_query, k=k, fetch_k=10, lambda_mult=0.75, filter=where)

        vector_context = ""
        for answer in docs:
            logging.debug("--------------------------------------------------------------------")
            logging.debug(answer.page_content)
            vector_context = vector_context + answer.page_content
        logging.info("--------------------------------------------------------------------")
        vector_context = vector_context.replace("\n\n", "\n")
        logging.info(vector_context)


def main(
    query: str,
    k: int,
    collection_name: str,
    persist_directory: str,
    embeddings_type: str,
    search_type: str,
) -> None:
    prompt_template_dir = getenv("PROMPT_TEMPLATE_DIRECTORY")
    refined_query_data = get_refined_query_data(query, prompt_template_dir)
    test_embeddings(
        refined_query_data[0],
        refined_query_data[1],
        k,
        collection_name,
        persist_directory,
        embeddings_type,
        search_type,
    )


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")

    # Add arguments
    parser.add_argument(
        "--query",
        type=str,
        default="What have Sarah and John Connor have to do with Cyberdyne and Miles Dyson?",
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
    parser.add_argument(
        "--embeddings-type",
        type=str,
        default="spacy",
        help="The chosen embeddings type",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="mmr",
        help="The chosen search type. mmr or similarity",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        query=args.query,
        k=args.k,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        embeddings_type=args.embeddings_type,
        search_type=args.search_type,
    )
