"""
Basics
------

:mod:`textacy.extract.basics`: Extract basic components from a document or sentence
via spaCy, with bells and whistles for filtering the results.
"""

import operator
from collections.abc import Callable, Collection, Iterable
from functools import partial

from cytoolz import itertoolz
from spacy.parts_of_speech import DET
from spacy.tokens import Span

from document_parsing.utils import constants, errors, types, utils


def ngrams(
    doclike: types.DocLike,
    n: int | Collection[int],
    *,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
    include_pos: str | Collection[str] | None = None,
    exclude_pos: str | Collection[str] | None = None,
    min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of n-grams (``n`` consecutive tokens) from a spaCy
    ``Doc`` or ``Span``, for one or multiple ``n`` values, optionally filtering n-grams
    by the types and parts-of-speech of the constituent tokens.

    Args:
        doclike
        n: Number of tokens included per n-gram; for example, ``2`` yields bigrams
            and ``3`` yields trigrams. If multiple values are specified, then the
            collections of n-grams are concatenated together; for example, ``(2, 3)``
            yields bigrams and then trigrams.
        filter_stops: If True, remove ngrams that start or end with a stop word.
        filter_punct: If True, remove ngrams that contain any punctuation-only tokens.
        filter_nums: If True, remove ngrams that contain any numbers
            or number-like tokens (e.g. 10, 'ten').
        include_pos: Remove ngrams if any constituent tokens' part-of-speech tags
            ARE NOT included in this param.
        exclude_pos: Remove ngrams if any constituent tokens' part-of-speech tags
            ARE included in this param.
        min_freq: Remove ngrams that occur in ``doclike`` fewer than ``min_freq`` times

    Yields:
        Next ngram from ``doclike`` passing all specified filters, in order of appearance
        in the document.

    Raises:
        ValueError: if any ``n`` < 1
        TypeError: if ``include_pos`` or ``exclude_pos`` is not a str, a set of str,
            or a falsy value

    Note:
        Filtering by part-of-speech tag uses the universal POS tag set; for details,
        check spaCy's docs: https://spacy.io/api/annotation#pos-tagging
    """
    ns_: tuple[int, ...] = utils.to_tuple(n)
    if any(n_ < 1 for n_ in ns_):
        msg = "n must be greater than or equal to 1"
        raise ValueError(msg)

    ngrams_: Iterable[Span]
    for n_ in ns_:
        ngrams_ = (doclike[i : i + n_] for i in range(len(doclike) - n_ + 1))
        ngrams_ = (ng for ng in ngrams_ if not any(w.is_space for w in ng))
        if filter_stops is True:
            ngrams_ = (ng for ng in ngrams_ if not ng[0].is_stop and not ng[-1].is_stop)
        if filter_punct is True:
            ngrams_ = (ng for ng in ngrams_ if not any(w.is_punct for w in ng))
        if filter_nums is True:
            ngrams_ = (ng for ng in ngrams_ if not any(w.like_num for w in ng))
        if include_pos:
            include_pos_: set[str] = {pos.upper() for pos in utils.to_set(include_pos)}
            ngrams_ = (ng for ng in ngrams_ if all(w.pos_ in include_pos_ for w in ng))
        if exclude_pos:
            exclude_pos_: set[str] = {pos.upper() for pos in utils.to_set(exclude_pos)}
            ngrams_ = (ng for ng in ngrams_ if not any(w.pos_ in exclude_pos_ for w in ng))
        if min_freq > 1:
            ngrams_ = list(ngrams_)
            freqs = itertoolz.frequencies(ng.text.lower() for ng in ngrams_)
            ngrams_ = (ng for ng in ngrams_ if freqs[ng.text.lower()] >= min_freq)

        yield from ngrams_


def entities(
    doclike: types.DocLike,
    *,
    include_types: str | Collection[str] | None = None,
    exclude_types: str | Collection[str] | None = None,
    drop_determiners: bool = True,
    min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of named entities (PERSON, ORG, LOC, etc.) from
    a ``Doc``, optionally filtering by entity types and frequencies.

    Args:
        doclike
        include_types: Remove entities whose type IS NOT
            in this param; if "NUMERIC", all numeric entity types ("DATE",
            "MONEY", "ORDINAL", etc.) are included
        exclude_types: Remove entities whose type IS
            in this param; if "NUMERIC", all numeric entity types ("DATE",
            "MONEY", "ORDINAL", etc.) are excluded
        drop_determiners: Remove leading determiners (e.g. "the")
            from entities (e.g. "the United States" => "United States").

            .. note:: Entities from which a leading determiner has been removed
               are, effectively, *new* entities, and not saved to the ``Doc``
               from which they came. This is irritating but unavoidable, since
               this function is not meant to have side-effects on document state.
               If you're only using the text of the returned spans, this is no
               big deal, but watch out if you're counting on determiner-less
               entities associated with the doc downstream.

        min_freq: Remove entities that occur in ``doclike`` fewer
            than ``min_freq`` times

    Yields:
        Next entity from ``doclike`` passing all specified filters in order of appearance
        in the document

    Raises:
        TypeError: if ``include_types`` or ``exclude_types`` is not a str, a set of
            str, or a falsy value
    """
    ents = doclike.ents

    include_types = _parse_ent_types(include_types, "include")
    exclude_types = _parse_ent_types(exclude_types, "exclude")
    if include_types:
        if isinstance(include_types, str):
            ents = (ent for ent in ents if ent.label_ == include_types)
        elif isinstance(include_types, set | frozenset | list | tuple):
            ents = (ent for ent in ents if ent.label_ in include_types)
    if exclude_types:
        if isinstance(exclude_types, str):
            ents = (ent for ent in ents if ent.label_ != exclude_types)
        elif isinstance(exclude_types, set | frozenset | list | tuple):
            ents = (ent for ent in ents if ent.label_ not in exclude_types)
    if drop_determiners is True:
        ents = (
            ent if ent[0].pos != DET else Span(ent.doc, ent.start + 1, ent.end, label=ent.label, vector=ent.vector)
            for ent in ents
        )
    if min_freq > 1:
        ents = list(ents)  # type: ignore
        freqs = itertoolz.frequencies(ent.text.lower() for ent in ents)
        ents = (ent for ent in ents if freqs[ent.text.lower()] >= min_freq)

    yield from ents


def _parse_ent_types(ent_types: str | Collection[str] | None, which: str) -> str | set[str] | None:
    if not ent_types:
        return None
    elif isinstance(ent_types, str):
        ent_types = ent_types.upper()
        # replace the shorthand numeric case by its corresponding constant
        if ent_types == "NUMERIC":
            return constants.NUMERIC_ENT_TYPES
        else:
            return ent_types
    elif isinstance(ent_types, set | frozenset | list | tuple):
        ent_types = {ent_type.upper() for ent_type in ent_types}
        # again, replace the shorthand numeric case by its corresponding constant
        # and include it in the set in case other types are specified
        if any(ent_type == "NUMERIC" for ent_type in ent_types):
            return ent_types.union(constants.NUMERIC_ENT_TYPES)
        else:
            return ent_types
    else:
        raise TypeError(errors.type_invalid_msg(f"{which}_types", type(ent_types), [str | Collection[str]] | None))


def noun_chunks(doclike: types.DocLike, *, drop_determiners: bool = True, min_freq: int = 1) -> Iterable[Span]:
    """
    Extract an ordered sequence of noun chunks from a spacy-parsed doc, optionally
    filtering by frequency and dropping leading determiners.

    Args:
        doclike
        drop_determiners: Remove leading determiners (e.g. "the")
            from phrases (e.g. "the quick brown fox" => "quick brown fox")
        min_freq: Remove chunks that occur in ``doclike`` fewer than ``min_freq`` times

    Yields:
        Next noun chunk from ``doclike`` in order of appearance in the document
    """
    ncs: Iterable[Span]
    ncs = doclike.noun_chunks
    if drop_determiners is True:
        ncs = (nc if nc[0].pos != DET else nc[1:] for nc in ncs)
    if min_freq > 1:
        ncs = list(ncs)
        freqs = itertoolz.frequencies(nc.text.lower() for nc in ncs)
        ncs = (nc for nc in ncs if freqs[nc.text.lower()] >= min_freq)

    yield from ncs


def terms(
    doclike: types.DocLike,
    *,
    ngs: int | Collection[int] | types.DocLikeToSpans | None = None,
    ents: bool | types.DocLikeToSpans | None = None,
    ncs: bool | types.DocLikeToSpans | None = None,
    dedupe: bool = True,
) -> Iterable[Span]:
    """
    Extract one or multiple types of terms -- ngrams, entities, and/or noun chunks --
    from ``doclike`` as a single, concatenated collection, with optional deduplication
    of spans extracted by more than one type.

    .. code-block:: pycon

        >>> extract.terms(doc, ngs=2, ents=True, ncs=True)
        >>> extract.terms(doc, ngs=lambda doc: extract.ngrams(doc, n=2))
        >>> extract.terms(doc, ents=extract.entities)
        >>> extract.terms(doc, ents=partial(extract.entities, include_types="PERSON"))

    Args:
        doclike
        ngs: N-gram terms to be extracted.
            If one or multiple ints, :func:`textacy.extract.ngrams(doclike, n=ngs)` is
            used to extract terms; if a callable, ``ngs(doclike)`` is used to extract
            terms; if None, no n-gram terms are extracted.
        ents: Entity terms to be extracted.
            If True, :func:`textacy.extract.entities(doclike)` is used to extract terms;
            if a callable, ``ents(doclike)`` is used to extract terms;
            if None, no entity terms are extracted.
        ncs: Noun chunk terms to be extracted.
            If True, :func:`textacy.extract.noun_chunks(doclike)` is used to extract
            terms; if a callable, ``ncs(doclike)`` is used to extract terms;
            if None, no noun chunk terms are extracted.
        dedupe: If True, deduplicate terms whose spans are extracted by multiple types
            (e.g. a span that is both an n-gram and an entity), as identified by
            identical (start, stop) indexes in ``doclike``; otherwise, don't.

    Returns:
        Next term from ``doclike``, in order of n-grams then entities then noun chunks,
        with each collection's terms given in order of appearance.

    Note:
        This function is *not* to be confused with keyterm extraction, which leverages
        statistics and algorithms to quantify the "key"-ness of terms before returning
        the top-ranking terms. There is no such scoring or ranking here.

    See Also:
        - :func:`textacy.extact.ngrams()`
        - :func:`textacy.extact.entities()`
        - :func:`textacy.extact.noun_chunks()`
        - :mod:`textacy.extact.keyterms`
    """
    extractors = _get_extractors(ngs, ents, ncs)
    terms_ = itertoolz.concat(extractor(doclike) for extractor in extractors)
    if dedupe is True:
        terms_ = itertoolz.unique(terms_, lambda span: (span.start, span.end))
    yield from terms_


def _get_extractors(ngs, ents, ncs) -> list[types.DocLikeToSpans]:
    all_extractors = [
        _get_ngs_extractor(ngs),
        _get_ents_extractor(ents),
        _get_ncs_extractor(ncs),
    ]
    extractors = [extractor for extractor in all_extractors if extractor is not None]
    if not extractors:
        msg = "at least one term extractor must be specified"
        raise ValueError(msg)
    else:
        return extractors


def _get_ngs_extractor(ngs) -> types.DocLikeToSpans | None:
    if ngs is None:
        return None
    elif callable(ngs):
        return ngs
    elif isinstance(ngs, int) or (isinstance(ngs, Collection) and all(isinstance(ng, int) for ng in ngs)):
        return partial(ngrams, n=ngs)
    else:
        raise TypeError()


def _get_ents_extractor(ents) -> types.DocLikeToSpans | None:
    if ents is None:
        return None
    elif callable(ents):
        return ents
    elif isinstance(ents, bool):
        return entities
    else:
        raise TypeError()


def _get_ncs_extractor(ncs) -> types.DocLikeToSpans | None:
    if ncs is None:
        return None
    elif callable(ncs):
        return ncs
    elif isinstance(ncs, bool):
        return noun_chunks
    else:
        raise TypeError()


def terms_to_strings(
    terms: Iterable[types.SpanLike],
    by: str | Callable[[types.SpanLike], str],
) -> Iterable[str]:
    """
    Transform a sequence of terms as spaCy ``Token`` s or ``Span`` s into strings.

    Args:
        terms
        by: Method by which terms are transformed into strings.
            If "orth", terms are represented by their text exactly as written;
            if "lower", by the lowercased form of their text;
            if "lemma", by their base form w/o inflectional suffixes;
            if a callable, must accept a ``Token`` or ``Span`` and return a string.

    Yields:
        Next term in ``terms``, as a string.
    """
    terms_: Iterable[str]
    if by == "lower":
        terms_ = (term.text.lower() for term in terms)
    elif by in ("lemma", "orth"):
        by_ = operator.attrgetter(f"{by}_")
        terms_ = (by_(term) for term in terms)
    elif callable(by):
        terms_ = (by(term) for term in terms)
    else:
        raise ValueError(errors.value_invalid_msg("by", by, {"orth", "lower", "lemma", Callable}))
    yield from terms_
