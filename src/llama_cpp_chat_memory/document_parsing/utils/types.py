"""
:mod:`textacy.types`: Definitions for common object types used throughout the package.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

PathLike = str | Path

DocLike = Doc | Span
SpanLike = Span | Token
DocLikeToSpans = Callable[[DocLike], Iterable[Span]]

LangLikeInContext = PathLike | Language | Callable[[str], str] | Callable[[str], Path] | Callable[[str], Language]


# typed equivalent to Record = collections.namedtuple("Record", ["text", "meta"])
class Record(NamedTuple):
    text: str
    meta: dict


DocData = str | Record | Doc
