import logging
import sys

import chainlit as cl
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

sys.path.append(".")
sys.path.append("..")
from llama_cpp_chat_memory import (
    AVATAR_IMAGE,
    CHARACTER_NAME,
    LLM_MODEL,
    PROMPT,
    RETRIEVER,
    USE_AVATAR_IMAGE,
    getenv,
)

logging.basicConfig(format="%(message)s", encoding="utf-8", level=logging.INFO)


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": CHARACTER_NAME}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    if USE_AVATAR_IMAGE:
        await cl.Avatar(name=CHARACTER_NAME, path=AVATAR_IMAGE, size="large").send()

    chain = ConversationChain(
        prompt=PROMPT,
        llm=LLM_MODEL,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=int(getenv("BUFFER_K")), human_prefix="User", ai_prefix=CHARACTER_NAME),
    )

    cl.user_session.set("llm_chain", chain)


async def get_answer(message, llm_chain: ConversationChain, callback):
    logging.info(message)
    docs = RETRIEVER.similarity_search_with_score(query=message, k=int(getenv("VECTOR_K")))
    logging.info(f"There are {RETRIEVER._collection.count()} documents in the collection")
    vector_context = ""
    for answer in docs:
        vector_context = vector_context + answer[0].page_content

    logging.info(vector_context)
    llm_chain.prompt = llm_chain.prompt.partial(vector_context=vector_context)

    result = await cl.make_async(llm_chain)(message, callbacks=[callback], return_only_outputs=True)
    return result["response"]


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Assistant", "AI", CHARACTER_NAME],
    )
    result = await get_answer(message=message.content, llm_chain=llm_chain, callback=cb)
    await cl.Message(content=result).send()
