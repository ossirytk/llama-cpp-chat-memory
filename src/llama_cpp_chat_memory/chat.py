from llama_cpp_chat_memory import *
from langchain import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import chainlit as cl


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": CHARACTER_NAME}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    global INIT
    USE_AVATAR_IMAGE

    if USE_AVATAR_IMAGE:
        await cl.Avatar(name=CHARACTER_NAME, path=AVATAR_IMAGE, size="large").send()

    llm_chain = ConversationChain(prompt=PROMPT, llm=LLM_MODEL, memory=ConversationBufferWindowMemory(k=10))
    cl.user_session.set("llm_chain", llm_chain)
    INIT = True


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    cb.answer_reached = True
    res = await cl.make_async(llm_chain)(message, callbacks=[cb])
