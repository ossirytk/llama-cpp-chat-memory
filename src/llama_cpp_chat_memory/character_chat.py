import chainlit as cl
from chainlit.input_widget import Select
from conversation_manager import ConveresationManager

conversation_manager = ConveresationManager()


@cl.author_rename
def rename(orig_author: str):
    # Renames chatbot to whatever the current character card name is
    rename_dict = {"Chatbot": conversation_manager.get_character_name()}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            Select(
                id="prompt_template_options",
                label="Prompt Templates",
                values=conversation_manager.get_prompt_templates(),
                initial_index=conversation_manager.get_prompt_template_index(),
            ),
            Select(
                id="context_collection",
                label="Context Collection",
                values=conversation_manager.get_context_collections(),
                initial_index=conversation_manager.get_context_index(),
            ),
            Select(
                id="mex_collection",
                label="Mex. Collection",
                values=conversation_manager.get_mes_collections(),
                initial_index=conversation_manager.get_mes_index(),
            ),
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings: dict[str, str]):
    conversation_manager.update_settings(settings)


@cl.on_message
async def main(message: cl.Message):

    result: cl.Message = await conversation_manager.ask_question(message)
    await result.send()
