You will need hatch to run this project. You can install hatch with pipx. See [Hatch](https://pypi.org/project/hatch/) and [Pipx](https://pipx.pypa.io/latest/installation/). The commands here are for windows powershell. If you use another shell, you'll have to change things as needed.
```
pip install pipx
pipx install hatch
```
Then from the repo root folder run.
```
hatch shell chat
cd .\src\llama_cpp_chat_memory\
python -m spacy download en_core_web_lg
playwright install
```

You will need spacy models for text embeddings if you do not use llama-cpp embeddings. Playwright is used by the old webscrape scripts. These are not needed for running the chatbot itself.</BR>

You also might want to run llama-cpp with gpu acceleration like cuda. See [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for specifics. Then run:
```
$env:FORCE_CMAKE=1
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-deps
```

Note that this example is for powershell and for the latest llama-cpp-python. You will need to change the command based on the terminal and the llama-cpp-python version.</BR>

Get a gguf model from a site like
[The Bloke](https://huggingface.co/models?sort=modified&search=theBloke+gguf)
and a character card and lorebooks from a site like [Character hub](https://www.characterhub.org/) or make your own with [character editor](https://character-tools.srjuggernaut.dev/character-editor)<BR>

Change the .env_test to .env and make sure that the correct folders exist.</BR>

You can set the collection to "" and try the chatbot by running:
```
chainlit run character_chat.py
```
If you want to create memory then see more details below.