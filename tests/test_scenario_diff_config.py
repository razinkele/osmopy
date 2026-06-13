"""Unit tests for _classify_config_diffs (Scenario Diff config-diff panel)."""

from __future__ import annotations

from ui.pages.scenario_diff import _classify_config_diffs


def test_changed_row_both_present():
    out = _classify_config_diffs([{"key": "a", "value_a": "1", "value_b": "2"}])
    assert out == [{"key": "a", "value_a": "1", "value_b": "2", "change": "changed"}]


def test_added_row_value_a_none():
    out = _classify_config_diffs([{"key": "a", "value_a": None, "value_b": "2"}])
    assert out[0]["change"] == "added"


def test_removed_row_value_b_none():
    out = _classify_config_diffs([{"key": "a", "value_a": "1", "value_b": None}])
    assert out[0]["change"] == "removed"


def test_empty_value_strings_are_changed_not_added_or_removed():
    # "" is a present value, not a missing key — only None drives added/removed.
    out = _classify_config_diffs([{"key": "a", "value_a": "", "value_b": "x"}])
    assert out[0]["change"] == "changed"


def test_sort_changed_then_added_then_removed_alpha_within_group():
    diffs = [
        {"key": "z_changed", "value_a": "1", "value_b": "2"},
        {"key": "a_added", "value_a": None, "value_b": "2"},
        {"key": "m_removed", "value_a": "1", "value_b": None},
        {"key": "a_changed", "value_a": "1", "value_b": "2"},
    ]
    out = _classify_config_diffs(diffs)
    assert [r["key"] for r in out] == ["a_changed", "z_changed", "a_added", "m_removed"]
    assert [r["change"] for r in out] == ["changed", "changed", "added", "removed"]


def test_order_independence_scrambled_input():
    # Deliberately scrambled (removed, changed, added; keys not alphabetical).
    scrambled = [
        {"key": "y_removed", "value_a": "1", "value_b": None},
        {"key": "b_changed", "value_a": "1", "value_b": "2"},
        {"key": "x_added", "value_a": None, "value_b": "2"},
        {"key": "a_changed", "value_a": "3", "value_b": "4"},
    ]
    out = _classify_config_diffs(scrambled)
    assert [r["key"] for r in out] == ["a_changed", "b_changed", "x_added", "y_removed"]


def test_empty_input_returns_empty_list():
    assert _classify_config_diffs([]) == []


def test_rows_preserve_key_and_values_verbatim():
    out = _classify_config_diffs([{"key": "species.linf.sp0", "value_a": "0.5", "value_b": "0.7"}])
    assert out[0]["key"] == "species.linf.sp0"
    assert out[0]["value_a"] == "0.5"
    assert out[0]["value_b"] == "0.7"
