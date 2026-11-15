"""Multi-value (per-stage array) config entries, recognised the way the engine parses them.

Which separators make an array depends on the KEY, not just the value:

- ``;`` marks an array on any key (the config writer round-trips it, and every array-capable
  engine parser accepts it).
- ``,`` marks an array only on fields flagged ``array=True`` in the schema: the ones the engine
  reads with ``_parse_floats`` (``re.split(r"[;,]")``), i.e. size ratios and stage thresholds.
  EEC full writes "2.3;1.8", Benguela "15, 5"; both are two-stage values. On any other key a ','
  is not a separator (the engine reads it with ``float()`` and fails), so "0,2" on a scalar key
  is a decimal-comma typo that the validator must keep reporting.

Every place that handles config values outside the engine (the validator, the form tabs, the
species and resource tables, the sync guard, the Advanced editor) must use this one test, or an
array is validated as one bad number, or flattened to a single value the engine then refuses.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osmose.schema.base import OsmoseField

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


def is_multi_value(value: object, field: OsmoseField | None = None) -> bool:
    """True when the engine would split ``value`` (the value of ``field``) into several values.

    Without a field only ';' counts, since a ',' is a separator only for array-capable fields.
    """
    if not isinstance(value, str):
        return False
    if ";" in value:
        return True
    if "," not in value or field is None or not field.array:
        return False
    parts = value.split(",")
    return all(_numeric_or_unset(p) for p in parts) and sum(1 for p in parts if p.strip()) >= 2


def split_multi_value(
    value: str, field: OsmoseField | None = None, keep_empty: bool = False
) -> list[str]:
    """The parts of a multi-value entry of ``field``, stripped: split on ';' and ',' for an
    array-capable field, on ';' alone otherwise. Empty parts are dropped (as the engine drops
    them) unless ``keep_empty``, which keeps each part's position for error messages."""
    if field is not None and field.array:
        raw = _SEPARATORS.split(value)
    else:
        raw = value.split(";")
    parts = [p.strip() for p in raw]
    return parts if keep_empty else [p for p in parts if p]
