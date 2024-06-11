# import spacy_stanza
# import stanza
from spacy_conll import init_parser

# stanza.download("en")

# Initialize the pipeline
# nlp = spacy_stanza.load_pipeline("en")
nlp = init_parser("en", "stanza")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
# for token in doc:
#    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
# print(doc.ents)
print(doc._.conll_pd)
