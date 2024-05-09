"""
spaCy Utils
-----------

:mod:`textacy.spacier.utils`: Helper functions for working with / extending spaCy's
core functionality.
"""

import pathlib

from spacy.language import Language

from document_parsing.spacier import core
from document_parsing.utils import errors, types


def resolve_langlikeincontext(text: str, lang: types.LangLikeInContext) -> Language:
    if isinstance(lang, Language):
        return lang
    elif isinstance(lang, str | pathlib.Path):
        return core.load_spacy_lang(lang)
    elif callable(lang):
        return resolve_langlikeincontext(text, lang(text))
    else:
        raise TypeError(errors.type_invalid_msg("lang", type(lang), types.LangLikeInContext))
