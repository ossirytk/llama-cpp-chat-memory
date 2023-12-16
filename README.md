# llama-cpp-chat-memory
This project is a llama-cpp character AI chatbot using tavern or V2 character cards and ChromaDB for character memory. You can use it as just a normal chatbot by setting the collection to "" in setting. 

There are several scripts for parsing json lorebooks, pdt, textfiles and scarping web pages for the memory content. Also included are scripts for parsing metadata from documents automatically. 

This project mainly serves as an example of langchain chatbot and memory, and is a template for further langchain projects. Uses chainlit as a dropin UI for the chatbot so there is basically no ui code. 

### Quickstart
You will need venv, pipenv or hatch to run this project. Hatch is recommended. You can install hatch with pipx
>pip install pipx</BR>
pipx install hatch</BR>

Then from the repo root folder run
>hatch shell chat</BR>
cd .\src\llama_cpp_chat_memory\
python -m spacy download en_core_web_lg</BR>
playwright install</BR>

You will need playwright for webscraping and the spacy models fot text embeddings if you do not use llama-cpp embeddings.</BR>

You might want to run llama-cpp with gpu acceleration like cuda. See [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for specifics. Then run:
>$env:FORCE_CMAKE=1</BR>
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"</BR>
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-deps</BR>

Note that this example is for powershell and for the latest llama-cpp-python. You will need to change the command based on the terminal and the llama-cpp-python version.</BR>

Get a gguf model from a site like
[The Bloke](https://huggingface.co/models?sort=modified&search=theBloke+gguf)
and a character card and lorebooks from a site like [Chub.ai](https://chub.ai/) or make your own with [character editor](https://zoltanai.github.io/character-editor/)<BR>

Change the .env_test to .env and make sure that the correct folders exist.</BR>

You can set the collection to "" and try the chatbot by running:
>chainlit run character_chat.py</BR>

If you want to create memory then see more details below.

### Prompt Support
Supports alpaca and mistral text prompts, v2 and tavern style json and yaml files and V2 and tavern png cards. Avatar images need to be in the same folder as the prompt file. V2 and Tavern png files get a copy of the image without exif data in the project temp file.

### Card Format
See [character editor](https://zoltanai.github.io/character-editor/).<BR>
There are two example cards included 'Skynet', ['Harry Potter'](https://chub.ai/characters/potato7295/harry-potter-bffe8945) and ['Bronya Zaychik'](https://chub.ai/characters/Mafiuz/bronya-zaychik-silverwing-n-ex-926eb8d4)<BR>
'name' : 'char_name'<br>
The name for the ai character. When using json or yaml, this is expected to correspond to avatar image. name.png or name.jpg.<br>
'description' : 'char_persona'<br>
The description for the character personality. Likes, dislikes, personality traits.<br>
'scenario' : 'world_scenario'<br>
Description of the scenario. This roughly corresponds to things like "You are a hr customer service having a discussion with a customer. Always be polite". etc.<br>
'mes_example' : 'example_dialogue'<br>
Example dialogue. The AI will pick answer patterns based on this<br>
'first_mes' : 'char_greeting'<br>
A landing page for the chat. This will not be included in the prompt.

### Configs
See the configuration params in .env.example file
VECTOR_K is the value for vector storage documents for how many documents should be returned. You might need to change this based on your context and vector store chunk size. BUFFER_K is the size for conversation buffer. The prompt will include last K qustion answer pairs. Having large VECTOR_K and BUFFER_K can overfill the prompt. The default character card is Skynet_V2.png. This is just a basic template. You can check and edit the content in [character editor](https://zoltanai.github.io/character-editor/)

The available embeddings are llama,spacy and hugginface. Make sure that the config for the chat matches the embeddings that were used to create the chroma collection. 

### Preparing the env
You will need a llama model that is compatible with llama-cpp. See models in HuggingFace by [The Bloke](https://huggingface.co/models?sort=modified&search=theBloke+gguf)<BR>
You might want to build with cuda support. <BR>
You need to pass FORCE_CMAKE=1 and CMAKE_ARGS="-DLLAMA_CUBLAS=on" to env variables. This is the powershell syntax. Use whatever syntax your shell uses to set env variables<BR>
You need to download models is you use NER parsing or spacy sentence transformers. The default model is en_core_web_lg. See available models at [Spacy Models](https://spacy.io/usage/models)<BR>
>python -m spacy download en_core_web_sm<BR>
python -m spacy download en_core_web_md<BR>
python -m spacy download en_core_web_lg<BR>

For installing dependencies in the virtual envs with pipenv
>pipenv install - requirements.txt<BR>

For installing dependencies in the virtual envs with hatch
>hatch env create<BR>

Copy the .env_test to .env and set directories and model settings
NOTE: Setting collection to "" will disable chroma fetching and you will get a normal character chatbot.

### Running the env 
>pipenv shell<BR>
(optional for cuda support)$env:FORCE_CMAKE=1<BR>
(optional for cuda support)$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"<BR>
(optional for cuda support)pip install llama-cpp-python==VERSION --force-reinstall --upgrade --no-cache-dir --no-deps<BR>
cd src\llama_cpp_langchain_chat<BR>

OR<BR>
>hatch shell chat<BR>
(optional for cuda support)$env:FORCE_CMAKE=1<BR>
(optional for cuda support)$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"<BR>
(optional for cuda support)pip install llama-cpp-python==VERSION --force-reinstall --upgrade --no-cache-dir --no-deps<BR>
cd src\llama_cpp_langchain_chat<BR>

### Web Scparing
You can scrape web pages to text documents in order to use them as documents for chroma. The web scraping uses playwright and requires that the web engines are installed. After starting the virtual env run:</BR>

> playwright install

The web scraping is prepared with config files in web_scrape_configs folder. The format is in json. See the example files for the specfics. The current impelemntation is unoptimized, so use with caution for a large number of pages.</BR>

To run the scrape run:
>python -m document-parsing.web_craper</BR>

See --help for params

### Creating embeddings
The embeddings creation uses env setting for threading and cuda
Use --help for basic instructions.<BR>
The parsing script will parse all txt, pdf or json files in the target directory. For json lorebooks a key_storage file will also be created for metadata filtering.<BR>
You need to download models for NER parsing. Textacy parses text files with Spacy sentence transformers to automatically generate keys for metadata filters. The default model is en_core_web_lg. See available models at [Spacy Models](https://spacy.io/usage/models)<BR>
>python -m spacy download en_core_web_sm<BR>
python -m spacy download en_core_web_md<BR>
python -m spacy download en_core_web_lg<BR>


You might want to play with the chunk size and overlap based on your text documents<BR>
The example documents include a txt file for skynet embeddings and json lorebooks for [Hogwarts](https://chub.ai/lorebooks/deadgirlz/hogwarts-legacy-lore-b819ccba) and [Honkai Impact](https://chub.ai/lorebooks/Zareh-Haadris/lorebook-honkai-impact-b1fcfc23)<BR>
The supported lorebook formats are chub inferred AgnAIstic and SillyTavern original source.
For pdf files there is a pdf file of short stories from Fyodor Dostoyevsky included The source is Internet Archive, the copy is in public domain. The pdf text quality is quite poor thought, so I recommend getting another file,

**!!Important!!.** You need to make sure that the documents, character_storage and key_storage folders exist.

NOTE: Textacy parsing will create a key file in key_storage that can be used by text parsing. Json files will create keys automatically if present in json file.
>python -m document_parsing.textacy_parsing --collection-name skynet --embeddings-type spacy<BR>
>python -m document_parsing.parse_text_documents --embeddings-type llama<BR>
>python -m document_parsing.parse_text_documents --collection-name skynet2 --embeddings-type spacy<BR>
>python -m document_parsing.parse_json_documents --embeddings-type spacy<BR>
>python -m document_parsing.test_embeddings  --collection-name skynet --query "Who is John Connor" --embeddings-type llama<BR>
>python -m document_parsing.test_embeddings  --collection-name skynet2 --query "Who is John Connor" --embeddings-type spacy<BR>
>python -m document_parsing.test_embeddings  --collection-name hogwarts --query "Who is Charles Rookwood'" --embeddings-type spacy<BR>

### Running the chatbot
>cd src\llama_cpp_langchain_chat<BR>
chainlit run character_chat.py<BR>

The chatbot should open in your browser<BR>

### Some examples 
![skynet01](/readme_pics/skynet01.png)
![skynet02](/readme_pics/skynet02.png)
![skynet03](/readme_pics/skynet03.png)
![skynet04](/readme_pics/skynet04.png)