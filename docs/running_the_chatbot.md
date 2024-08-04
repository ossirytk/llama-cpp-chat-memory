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
### Running the terminal chatbot
You can run the chatbot directly in the terminal without starting a web browser. The terminal script is a low effort way to debug the chatbot quickly.

```
cd src\llama_cpp_langchain_chat
python -m document_parsing.terminal_chatbot
```
### Avatar Images
Avatar images need to be stored in the .\public\avatars folder. Make sure that the folder exists. Character cards in png format will have a copy of the image data saved in the avatars folder automatically. If you copy an image manually, make sure that the filename matches the name is the character card and replace the whitespace in the name with underscores.
### Vector search
The search for relevant documents from chroma happens based on VECTOR_SORT_TYPE and VECTOR_K. The search will return VECTOR_K+4 closest matches and sorts by sort type before appending to vector_k. Default search simply sorts by distance. "bm25" sorts with the bm25 search algorithm. Fusion rank gets the combined results of both.
### Query metadata
The query is parsed for metadata using spacy. The metadata keys are used as a filter when searching the Chroma collections.