"""
:mod:`textacy.utils`: Variety of general-purpose utility functions for inspecting /
validating / transforming args and facilitating meta package tasks.
"""

from collections.abc import Iterable
from typing import (
    Any,
)

# a (text, metadata) 2-tuple
RECORD_LEN = 2


def is_record(obj: Any) -> bool:
    """Check whether ``obj`` is a "record" -- that is, a (text, metadata) 2-tuple."""
    if isinstance(obj, tuple) and len(obj) == RECORD_LEN and isinstance(obj[0], str) and isinstance(obj[1], dict):
        return True
    else:
        return False


def to_set(val: Any) -> set:
    """Cast ``val`` into a set, if necessary and possible."""
    if isinstance(val, set):
        return val
    elif isinstance(val, Iterable) and not isinstance(val, str | bytes):
        return set(val)
    else:
        return {val}


def to_tuple(val: Any) -> tuple:
    """Cast ``val`` into a tuple, if necessary and possible."""
    if isinstance(val, tuple):
        return val
    elif isinstance(val, Iterable) and not isinstance(val, str | bytes):
        return tuple(val)
    else:
        return (val,)
