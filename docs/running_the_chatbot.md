### Running the chatbot
To run the chatbot. You need to run the chat with the custom script instead of the chainlit run command.
The reason for this is the updates for the config files when switching character. 
These changes need to be done before calling chainlit.

If you call chainlit directly, the character name and avatar picture won't update.

Note: Currently something seems to be cached by chainlit. Until I find a way to clear the cache,
you need to call run_chat twice for changes to take effect.

```
cd src\llama_cpp_langchain_chat
python -m run_chat
```

The chatbot should open in your browser<BR>