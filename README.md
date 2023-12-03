# llama-cpp-chat-memory
Llamacpp chat with langchain and chainlit using vector stores as memory. This project mainly serves as a simple example of langchain chatbot and is a template for further langchain projects. Uses chainlit as a dropin UI chatbot so there is basically no ui code. 

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

### Preparing the env
You might want to build with cuda support. <BR>
You need to pass FORCE_CMAKE=1 and CMAKE_ARGS="-DLLAMA_CUBLAS=on" to env variables<BR>
>pipenv install - requirements.txt<BR>
pipenv shell<BR>
(optional for cuda support)$env:FORCE_CMAKE=1<BR>
(optional for cuda support)$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"<BR>
(optional for cuda support)pip install llama-cpp-python==VERSION --force-reinstall --upgrade --no-cache-dir --no-deps<BR>

NOTE: Setting collection to "" will disable chroma fetching and you will get a normal character chatbot.

### Running the env 
>pipenv shell<BR>
cd src\llama_cpp_langchain_chat<BR>

OR<BR>
>hatch env create<BR>
hatch shell chat<BR>
cd src\llama_cpp_langchain_chat<BR>

### Creating embeddings
The embeddings creation uses env setting for threading and cuda
Use --help for basic instructions.<BR>
The parsing script will parse all txt, pdf or json files in the target directory. For json lorebooks a key_storage file will also be created for metadata filtering.<BR>
You need to download models for NER parsing. Textacy parses text files with Spacy sentence transformers to automatically generate keys for metadata filters. The default model is en_core_web_lg<BR>
>python -m spacy download en_core_web_sm<BR>
python -m spacy download en_core_web_md<BR>
python -m spacy download en_core_web_lg<BR>


You might want to play with the chunk size and overlap based on your text documents<BR>
The example documents include a txt file for skynet embeddings and json lorebooks for [Hogwarts](https://chub.ai/lorebooks/deadgirlz/hogwarts-legacy-lore-b819ccba) and [Honkai Impact](https://chub.ai/lorebooks/Zareh-Haadris/lorebook-honkai-impact-b1fcfc23)<BR>
The supported lorebook formats are chub inferred AgnAIstic and SillyTavern original source.
For pdf files there is a pdf file of short stories from Fyodor Dostoyevsky included The source is Internet Archive, the copy is in public domain. The pdf text quality is quite poor thought, so I recommend getting another file,

**!!Important!!.** You need to make sure that the documents, character_storage and key_storage folders exist.

For parsing documents run these in the document_parsing directory. Textacy parsing will create a key file in key_storage that can be used by text parsing.
>python parse_text_documents.py<BR>

>python parse_json_documents.py<BR>

>python parse_pdf_documents.py<BR>

>python parse_pdf_documents.py<BR>

>python textacy_parsing.py<BR>

For parsing hogwarts.json lore book<BR>
> python .\parse_json_documents.py --collection-name hogwarts --data-directory .\documents\hogwarts\

### Testing embeddings
The embeddings test uses env setting for threading and cuda
>cd python test_embeddings.py

or
>python .\test_embeddings.py --collection-name hogwarts --query "Who is Charles Rookwood" --k 5

### Running the chatbot
>cd src\llama_cpp_langchain_chat<BR>
chainlit run character_chat.py<BR>

The chatbot should open in your browser<BR>

### Some examples 
![skynet01](/readme_pics/skynet01.png)
![skynet02](/readme_pics/skynet02.png)
![skynet03](/readme_pics/skynet03.png)
![skynet04](/readme_pics/skynet04.png)