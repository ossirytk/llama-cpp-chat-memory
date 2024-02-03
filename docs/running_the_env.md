### Running the env
You'll need to run all the commands inside the virtual env.
```
hatch shell chat
(optional for cuda support)$env:FORCE_CMAKE=1
(optional for cuda support)$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
(optional for cuda support)pip install llama-cpp-python==VERSION --force-reinstall --upgrade --no-cache-dir --no-deps
cd src\llama_cpp_langchain_chat
```