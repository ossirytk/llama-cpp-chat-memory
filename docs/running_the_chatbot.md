### Running the chatbot
To run the chatbot. You need to run the chat with the custom script instead of the chainlit run command.
The reason for this is the updates for the config files when switching character. 
These changes need to be done before calling chainlit.

If you call chainlit directly, the character name and avatar picture won't update.

Note: Currently something seems to be cached by chainlit. Until I find a way to clear the cache,
you need to call run_chat twice for changes to take effect.

Some browsers don't allow loading css file from local directories. For testing purposes there is a flask script to run a simple http server that serves stylesheets from the "static/" directory. You will need to run the flask server in another terminal instance.

```
cd src\llama_cpp_langchain_chat
python -m run_chat
```

The chatbot should open in your browser<BR>

Running flask
```
hatch shell chat
cd .\src\llama_cpp_chat_memory\
flask --app flask_web_server run
```
### Avatar Images
Avatar images need to be stored in the .\public\avatars folder. Make sure that the folder exists. Character cards in png format will have a copy of the image data saved in the avatars folder automatically. If you copy an image manually, make sure that the filename matches the name is the character card and replace the whitespace in the name with underscores.