### Preparing the env
You will need a llama model that is compatible with llama-cpp. See models in HuggingFace by [The Bloke](https://huggingface.co/models?sort=modified&search=theBloke+gguf)<BR>
You might want to build with cuda support. <BR>
You need to pass FORCE_CMAKE=1 and CMAKE_ARGS="-DLLAMA_CUBLAS=on" to env variables. This is the powershell syntax. Use whatever syntax your shell uses to set env variables<BR>
You need to download language models if you use NER parsing, embeddings or spacy sentence transformers. The default model is en_core_web_lg. See available models at [Spacy Models](https://spacy.io/usage/models)<BR>
Choose the preferred model size and type.
```
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

For installing dependencies in the virtual envs with hatch
```
hatch env create
```
Copy the .env_test to .env and set directories and model settings
NOTE: Setting collection to "" will disable chroma fetching and you will get a normal character chatbot.