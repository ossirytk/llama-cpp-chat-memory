import chainlit as cl
from conversation_manager import ConveresationManager

conversation_manager = ConveresationManager()


@cl.author_rename
def rename(orig_author: str):
    # Renames chatbot to whatever the current character card name is
    rename_dict = {"Chatbot": conversation_manager.get_character_name()}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    # Set the chatbot icon to character icon
    if conversation_manager.get_use_avatar_image():
        await cl.Avatar(
            name=conversation_manager.get_character_name(),
            path=conversation_manager.get_avatar_image_path(),
            size="large",
        ).send()


def get_answer(message, callback):
    answer = conversation_manager.ask_question(message, callback)
    return answer


@cl.on_message
async def main(message: str):
    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Assistant", "AI", conversation_manager.get_character_name()],
    )
    result = await cl.make_async(get_answer)(message=message.content, callback=cb)
    await cl.Message(content=result).send()
