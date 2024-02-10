### Named Entity Recognition(NER)
You can use textacy_parsing script for generating document metadata keys automatically. The scripts are a modified version of textacy code updated to run with the current spacy version. The script uses a spacy embeddings model to process a text document for a json metadata keyfile. The include positions are: "PROPN", "NOUN", "ADJ". The includes entities are: "PRODUCT", "EVENT", "FAC", "NORP", "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "WORK_OF_ART". For details see [Spacy linguistic features](https://spacy.io/usage/linguistic-features) and [Model NER labels](https://spacy.io/models/en). The instructions expect en model, but spacy supports a wide range of models.

You can create ner metadata list with
```
python -m document_parsing.textacy_parsing
```

Optional param         | Description
---------------------- | -------------
--data-directory       | The directory where your text files are stored. Default "./documents/skynet"
--collection-name      | The name of the collection Will be used as name and location for the keyfile. Default "skynet"
--key-storage          | The directory for the collection metadata keys. Default "./key_storage/"