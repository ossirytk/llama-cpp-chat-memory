### Running the env
You'll need to run all the commands inside the virtual env. Some browsers don't allow loading css file from local directories. For testing purposes there is a flask script to run a simple http server that serves stylesheets from the "static/" directory. You will need to run the flask server in another terminal instance.
```
hatch shell chat
(optional for cuda support)$env:FORCE_CMAKE=1
(optional for cuda support)$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
(optional for cuda support)pip install llama-cpp-python==VERSION --force-reinstall --upgrade --no-cache-dir --no-deps
cd src\llama_cpp_langchain_chat
```

Running flask
```
hatch shell chat
cd .\src\llama_cpp_chat_memory\
flask --app flask_web_server run
```