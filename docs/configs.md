### Configs
You can change the configuration settings in .env file.

The available embeddings are llama,spacy and hugginface. Make sure that the config for the chat matches the embeddings that were used to create the chroma collection. 

VECTOR_K is the value for vector storage documents for how many documents should be returned. You might need to change this based on your context and vector store chunk size. BUFFER_K is the size for conversation buffer. The prompt will include last K qustion answer pairs. Having large VECTOR_K and BUFFER_K can overfill the prompt. The default character card is Skynet_V2.png. This is just a basic template.

Config Field  | Description
------------- | -------------
MODEL_DIR     | The dir for the models
MODEL         | model_name.gguf
MODEL_TYPE    | alpaca/mistral
LAYERS        | Number of layers to offload to gpu
CHARACTER_CARD_DIR | The directory for chracter cards
CHARACTER_CARD | character_card.png/yaml/json
PERSIST_DIRECTORY | dir for chroma embeddings
PROMPT_TEMPLATE_DIRECTORY | Prompt template are stored here
PROMPT_TEMPLATE | question_generation_template.json
REPLACE_YOU | Replace references to "You" in card with "User"
KEY_STORAGE_DIRECTORY | dir for NER keys for chroma
USE_KEY_STORAGE | Use NER keys for Chroma metadata
COLLECTION | Chroma collection to use. "" to disable Chroma
QUERY_TYPE | Embeddings type. "mmr" or "similarity"
EMBEDDINGS_TYPE | llama/spacy/hugginface
FETCH_K | Fetch k closest embeddings for similiarity
LAMBDA_MULT | Lambda for Chroma
VECTOR_K | Fetch k closest embeddings for mmr
BUFFER_K | Buffer last k exchanges to conversation context
ROPE_CONTEXT | Rope context for rope scaling
N_CTX | Context size
USE_MAX_TOKENS | Use max tokens. True/False
MAX_TOKENS | Max tokens