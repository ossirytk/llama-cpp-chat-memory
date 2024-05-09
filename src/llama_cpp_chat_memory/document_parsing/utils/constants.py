"""
Collection of regular expressions and other (small, generally useful) constants.
"""

import re
from re import Pattern

NUMERIC_ENT_TYPES: set[str] = {
    "ORDINAL",
    "CARDINAL",
    "MONEY",
    "QUANTITY",
    "PERCENT",
    "TIME",
    "DATE",
}


RE_ALNUM: Pattern = re.compile(r"[^\W_]+")
