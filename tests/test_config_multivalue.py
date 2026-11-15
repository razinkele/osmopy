"""Multi-value config entries are recognised the way the engine parses them.

The engine splits per-stage arrays on BOTH ';' and ',' (``_parse_floats``, re.split(r"[;,]")), so
"2.3;1.8" (EEC full) and "15, 5" (Benguela) are both two-stage values. A comma counts only when
every part is numeric, so a path or a name that happens to contain a comma stays a single value.
"""

from __future__ import annotations

import pytest
from osmose.config.multivalue import is_multi_value, split_multi_value


@pytest.mark.parametrize(
    "value",
    ["2.3;1.8", "50;20", "15, 5", "300, 100", "1,2,3", "maps/a;b.csv", "0.5;null", "1, NaN"],
)
def test_multi_values(value):
    assert is_multi_value(value)


@pytest.mark.parametrize(
    "value", ["3.5", "", "null", "maps/a,b.csv", "Cod, Atlantic", "true", "1e-3", None]
)
def test_single_values(value):
    assert not is_multi_value(value)


def test_split_matches_the_engine_parser():
    """Same parts, in the same order, as osmose.engine.background._parse_floats."""
    from osmose.engine.background import _parse_floats

    for value in ("2.3;1.8", "15, 5", "1;2,3"):
        assert [float(p) for p in split_multi_value(value)] == _parse_floats(value)
