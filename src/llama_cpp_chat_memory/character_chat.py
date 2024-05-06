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


@cl.on_message
async def main(message: cl.Message):

    result: cl.Message = await conversation_manager.ask_question(message)
    await result.send()
