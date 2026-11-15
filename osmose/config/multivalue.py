"""Multi-value (per-stage array) config entries, recognised the way the engine parses them.

The engine splits per-stage arrays on BOTH ';' and ',' (``osmose.engine.background._parse_floats``:
``re.split(r"[;,]", value)``), so "2.3;1.8" (EEC full) and "15, 5" (Benguela) are both two-stage
values. Every place that handles config values outside the engine (the validator, the form tabs,
the species table, the sync guard, the Advanced editor, the larva-rate migration) must use the
same test, or an array is validated as one bad number, or flattened to a single value that the
engine then refuses.

A ';' always marks an array (the writer round-trips any ';'). A ',' marks one only when every part
is numeric or an unset sentinel, so a path or a name that contains a comma stays a single value.
"""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[;,]")
_UNSET = frozenset({"", "null", "none", "na", "nan"})


def _numeric_or_unset(part: str) -> bool:
    s = part.strip()
    if s.lower() in _UNSET:
        return True
    try:
        float(s)
    except ValueError:
        return False
    return True


def is_multi_value(value: object) -> bool:
    """True when ``value`` is a per-stage array the engine would split into several values."""
    if not isinstance(value, str):
        return False
    if ";" in value:
        return True
    if "," not in value:
        return False
    parts = value.split(",")
    return all(_numeric_or_unset(p) for p in parts) and any(p.strip() for p in parts)


def split_multi_value(value: str, keep_empty: bool = False) -> list[str]:
    """The parts of a multi-value entry, stripped. Empty parts are dropped (as the engine drops
    them) unless ``keep_empty``, which keeps each part's position for error messages."""
    parts = [p.strip() for p in _SEPARATORS.split(value)]
    return parts if keep_empty else [p for p in parts if p]
