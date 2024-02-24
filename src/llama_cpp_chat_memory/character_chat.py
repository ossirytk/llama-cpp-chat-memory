import sys

import chainlit as cl
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from numpy import float64

sys.path.append(".")
sys.path.append("..")
from llama_cpp_chat_memory import (  # noqa: E402
    ALL_KEYS,
    AVATAR_IMAGE,
    CHARACTER_NAME,
    CHAT_LOG,
    LLM_MODEL,
    PROMPT,
    QUESTION_REFINING_PROMPT,
    QUSTION_REFINING_METADATA_PROMPT,
    RETRIEVER,
    USE_AVATAR_IMAGE,
    getenv,
)


@cl.author_rename
def rename(orig_author: str):
    # Renames chatbot to whatever the current character card name is
    rename_dict = {"Chatbot": CHARACTER_NAME}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    # Set the chatbot icon to character icon
    if USE_AVATAR_IMAGE:
        await cl.Avatar(name=CHARACTER_NAME, path=AVATAR_IMAGE, size="large").send()

    # Use basic conversation chain with buffered conversation fistory
    chain = ConversationChain(
        prompt=PROMPT,
        llm=LLM_MODEL,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=int(getenv("BUFFER_K")), human_prefix="User", ai_prefix=CHARACTER_NAME),
    )

    cl.user_session.set("llm_chain", chain)


# async def get_answer(message, llm_chain: ConversationChain, callback):
def get_answer(message, llm_chain: ConversationChain, callback):
    CHAT_LOG.info(message)
    vector_context = ""
    if RETRIEVER:

        llm_chain_refine = LLMChain(prompt=QUESTION_REFINING_PROMPT, llm=LLM_MODEL)
        refined_query_result = llm_chain_refine.invoke(message)
        refined_query = refined_query_result["text"]
        llm_chain_refine.prompt = QUSTION_REFINING_METADATA_PROMPT
        metadata_result = llm_chain_refine.invoke(refined_query)
        metadata_query = metadata_result["text"]

        CHAT_LOG.info(f"Refined Query {refined_query}")
        CHAT_LOG.info(f"Query metadata {metadata_query}")
        # Currently Chroma has no "like" implementation so this is a case sensitive hack with uuids
        # There is also an issue when filter has only one item since "in" expects multiple items
        # With one item, just use a dict with "uuid", "filter"
        metadata_filter_list = []
        filter_list = {}
        if ALL_KEYS is not None:
            for item in ALL_KEYS.items():
                if item[1].lower() in metadata_query.lower():
                    filter_list[item[0]] = item[1]
                    metadata_filter_list.append({item[0]: {"$in": [item[1]]}})

        if len(filter_list) == 1:
            where = filter_list
        elif len(filter_list) > 1:
            where = {"$or": metadata_filter_list}
        else:
            where = None

        query_type = getenv("QUERY_TYPE")
        k = int(getenv("VECTOR_K"))
        CHAT_LOG.info(f"There are {RETRIEVER._collection.count()} documents in the collection")
        CHAT_LOG.info(f"Filter {where}")
        if query_type == "similarity":
            docs = RETRIEVER.similarity_search_with_score(query=refined_query, k=k, filter=where)
            for answer in docs:
                vector_context = vector_context + answer[0].page_content
        elif query_type == "mmr":
            docs = RETRIEVER.max_marginal_relevance_search(
                query=refined_query,
                k=k,
                fetch_k=int(getenv("FETCH_K")),
                lambda_mult=float64(getenv("LAMBDA_MULT")),
                filter=where,
            )
            for answer in docs:
                vector_context = vector_context + answer.page_content
        else:
            CHAT_LOG.error(f"{query_type} is not implemented")
            raise NotImplementedError()

        CHAT_LOG.info(vector_context)
    llm_chain.prompt = llm_chain.prompt.partial(vector_context=vector_context)

    # You will need make_async to actually make this run asynchronoysly
    # result = await cl.make_async(llm_chain.invoke())(message, callbacks=[callback])
    result = llm_chain.invoke(message, callbacks=[callback])
    return result["response"]


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Assistant", "AI", CHARACTER_NAME],
    )
    # result = await get_answer(message=message.content, llm_chain=llm_chain, callback=cb)
    # result = get_answer(message=message.content, llm_chain=llm_chain, callback=cb)
    result = await cl.make_async(get_answer)(message=message.content, llm_chain=llm_chain, callback=cb)
    await cl.Message(content=result).send()
