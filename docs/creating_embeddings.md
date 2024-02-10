### Creating embeddings
The embeddings creation uses env setting for threading and cuda. The Example documents are in the Documents folder. The scripts are in the documents_parsing folder. 
Use --help for basic instructions.<BR>
The parsing script will parse all txt, pdf or json files in the target directory. For json lorebooks a key_storage file will also be created for metadata filtering.<BR>
You need to download models for NER parsing. Textacy parses text files with Spacy sentence transformers to automatically generate keys for metadata filters. The default model is en_core_web_lg. See available models at [Spacy Models](https://spacy.io/usage/models)<BR>
```
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

You might want to play with the chunk size and overlap based on your text documents<BR>
The example documents include a txt file for skynet embeddings and json lorebooks for [Hogwarts](https://chub.ai/lorebooks/deadgirlz/hogwarts-legacy-lore-b819ccba) and [Honkai Impact](https://chub.ai/lorebooks/Zareh-Haadris/lorebook-honkai-impact-b1fcfc23)<BR>
The supported lorebook formats are chub inferred AgnAIstic and SillyTavern original source.
For pdf files there is a pdf file of short stories from Fyodor Dostoyevsky included The source is Internet Archive, the copy is in public domain. The pdf text quality is quite poor thought, so I recommend getting another file,

**!!Important!!.** You need to make sure that the documents, character_storage and key_storage folders exist.

Textacy parsing will use NER to parse keys from the document using sentence transformers. This keys can be used as Chroma metadata,
NOTE: Textacy parsing will create a key file in key_storage that can be used by text parsing. Json files will create keys automatically if present in json file.
```
python -m document_parsing.textacy_parsing --collection-name skynet --embeddings-type spacy
```
Parse the documents with
```
python -m document_parsing.parse_text_documents --embeddings-type llama
python -m document_parsing.parse_text_documents --collection-name skynet2 --embeddings-type spacy
python -m document_parsing.parse_json_documents --embeddings-type spacy
```

You can test the embeddings with
```
python -m document_parsing.test_embeddings  --collection-name skynet --query "Who is John Connor" --embeddings-type llama
python -m document_parsing.test_embeddings  --collection-name skynet2 --query "Who is John Connor" --embeddings-type spacy
python -m document_parsing.test_embeddings  --collection-name hogwarts --query "Who is Charles Rookwood'" --embeddings-type spacy
```

Optional param         | Description
---------------------- | -------------
--data-directory       | The directory where your text files are stored. Default "./documents/skynet"
--collection-name      | The name of the collection. Default "skynet"
--persist-directory    | The directory where you want to store the Chroma collection. Default "./character_storage/"
--key-storage          | The directory for the collection metadata keys Need to be created with textacy parsing. Default "./key_storage/"
--chunk-size           | The text chunk size for parsing. Default "1024"
--chunk-overlap        | The overlap for text chunks for parsing. Default "0"
--embeddings-type      | The chosen embeddings type. Default "spacy"