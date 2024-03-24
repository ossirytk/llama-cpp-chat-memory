### Named Entity Recognition(NER)
You can use textacy_parsing script for generating document metadata keys automatically. The scripts are a modified version of textacy code updated to run with the current spacy version. The script uses a spacy embeddings model to process a text document for a json metadata keyfile. The keys are parsed based on a config file in run_files/parse_configs/ner_types.json or run_files/parse_configs/ner_types_full.json. You can give your own config file if you want.

The available configs are

Ngrams        | Description
------------- | -------------
PROPN         | Proper Noun
NOUN          | Noun
ADJ           | Adjective
NNP           | Noun proper singular
NN            | Noun, singular or mass
AUX           | Auxiliary
VBZ           | Verb, 3rd person singular present
VERB          | Verb
ADP           | Adposition
SYM           | Symbol
NUM           | Numeral
CD            | Cardinal number
VBG           | verb, gerund or present participle
ROOT          | Root

Entities      | Description
------------- | -------------
FAC           | Buildings, airports, highways, bridges, etc.
NORP          | Nationalities or religious or political groups
GPE           | Countries, cities, states
PRODUCT       | Objects, vehicles, foods, etc. (not services)
EVENT         | Named hurricanes, battles, wars, sports events, etc.
PERSON        | People, including fictional
ORG           | Companies, agencies, institutions, etc.
LOC           | Non-GPE locations, mountain ranges, bodies of water
DATE          | Absolute or relative dates or periods
TIME          | Times smaller than a day
WORK_OF_ART   | Titles of books, songs, etc.

Extract type    | Description
--------------- | -------------
orth            | Terms are represented by their text exactly as written
lower           | Lowercased form of the text
lemma           | Base form w/o inflectional suffixes

For details see [Spacy linguistic features](https://spacy.io/usage/linguistic-features) and [Model NER labels](https://spacy.io/models/en). The instructions expect en model, but spacy supports a wide range of models. You can also specify Noun chunks. Noun chunk of 2 for example would create keys like "Yellow House" or "Blond Hair".



You can create ner metadata list with
```
python -m document_parsing.parse_ner
```

Optional param         | Description
---------------------- | -------------
--data-directory       | The directory where your text files are stored. Default "./run_files/documents/skynet"
--collection-name      | The name of the collection Will be used as name and location for the keyfile. Default "skynet"
--key-storage          | The directory for the collection metadata keys. Default "./run_files/key_storage/"