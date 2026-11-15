"""Multi-value config entries are recognised the way the engine parses them, which is key-aware.

';' marks an array on any key. ',' marks one only on fields flagged ``array=True`` (the keys the
engine reads with ``_parse_floats``: size ratios, stage thresholds). On any other key a ',' is a
decimal-comma typo the validator must report, not an array.
"""

from __future__ import annotations

import pytest

from osmose.config.multivalue import is_multi_value, split_multi_value
from osmose.schema import build_registry

_REG = build_registry()
ARRAY = _REG.match_field("predation.predprey.sizeratio.max.sp0")
SCALAR = _REG.match_field("species.linf.sp0")


def test_the_array_flag_is_set_exactly_where_the_engine_accepts_commas():
    flagged = sorted(f.key_pattern for f in _REG.all_fields() if f.array)
    assert flagged == [
        "predation.predprey.sizeratio.max.sp{idx}",
        "predation.predprey.sizeratio.min.sp{idx}",
        "predation.predprey.stage.threshold.sp{idx}",
    ]


@pytest.mark.parametrize("value", ["2.3;1.8", "maps/a;b.csv", "0.5;null", "25;5"])
@pytest.mark.parametrize("field", [ARRAY, SCALAR, None], ids=["array", "scalar", "nofield"])
def test_a_semicolon_marks_an_array_on_any_key(value, field):
    assert is_multi_value(value, field)


@pytest.mark.parametrize("value", ["15, 5", "300, 100", "1,2,3", "1, NaN"])
def test_a_comma_marks_an_array_only_on_an_array_field(value):
    assert is_multi_value(value, ARRAY)
    assert not is_multi_value(value, SCALAR)  # e.g. "0,2" is a decimal-comma typo there
    assert not is_multi_value(value, None)


@pytest.mark.parametrize("value", ["3.5", "", "null", "maps/a,b.csv", "2.3,", ",5", None])
def test_single_values_even_on_an_array_field(value):
    """A trailing or leading separator leaves ONE value for the engine, so not an array."""
    assert not is_multi_value(value, ARRAY)


def test_split_matches_the_engine_parser_on_an_array_field():
    """Same parts, in the same order, as osmose.engine.background._parse_floats."""
    from osmose.engine.background import _parse_floats

    for value in ("2.3;1.8", "15, 5", "1;2,3"):
        assert [float(p) for p in split_multi_value(value, ARRAY)] == _parse_floats(value)


def test_split_on_a_scalar_field_uses_semicolons_only():
    assert split_multi_value("1;2,3", SCALAR) == ["1", "2,3"]
