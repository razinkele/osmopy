"""Unit tests for the shared config-diff component (classify + render)."""

from ui.components.config_diff import classify_config_diffs, render_config_diff_table


def test_changed_when_both_present_and_differ():
    out = classify_config_diffs([{"key": "a", "value_a": "1", "value_b": "2"}])
    assert out[0]["change"] == "changed"


def test_added_when_value_a_none():
    out = classify_config_diffs([{"key": "a", "value_a": None, "value_b": "2"}])
    assert out[0]["change"] == "added"


def test_removed_when_value_b_none():
    out = classify_config_diffs([{"key": "a", "value_a": "1", "value_b": None}])
    assert out[0]["change"] == "removed"


def test_empty_string_is_changed_not_added_or_removed():
    out = classify_config_diffs([{"key": "a", "value_a": "", "value_b": "x"}])
    assert out[0]["change"] == "changed"


def test_sort_changed_added_removed_then_alpha():
    diffs = [
        {"key": "z_changed", "value_a": "1", "value_b": "2"},
        {"key": "a_removed", "value_a": "1", "value_b": None},
        {"key": "m_added", "value_a": None, "value_b": "2"},
        {"key": "a_changed", "value_a": "1", "value_b": "2"},
    ]
    out = classify_config_diffs(diffs)
    assert [r["key"] for r in out] == ["a_changed", "z_changed", "m_added", "a_removed"]


def test_deterministic_regardless_of_input_order():
    diffs = [
        {"key": "b", "value_a": "1", "value_b": "2"},
        {"key": "a", "value_a": "1", "value_b": "2"},
    ]
    assert classify_config_diffs(diffs) == classify_config_diffs(list(reversed(diffs)))


def test_mixed_case_keys_sort_case_sensitively():
    # Python str sort is case-sensitive: uppercase 'S' (83) sorts before lowercase 's' (115).
    diffs = [
        {"key": "species.linf", "value_a": "1", "value_b": "2"},
        {"key": "Species.Linf", "value_a": "1", "value_b": "2"},
    ]
    out = classify_config_diffs(diffs)
    assert [r["key"] for r in out] == ["Species.Linf", "species.linf"]


def test_empty_input():
    assert classify_config_diffs([]) == []


def test_render_returns_count_and_table():
    html = str(
        render_config_diff_table([{"key": "species.linf.sp0", "value_a": "0.5", "value_b": "0.7"}])
    )
    assert "1 differing config key" in html
    assert "<table" in html
    assert "species.linf.sp0" in html


def test_render_none_cell_shows_exactly_one_dash():
    html = str(render_config_diff_table([{"key": "a", "value_a": None, "value_b": "x"}]))
    assert html.count("—") == 1  # exactly the one None cell, nothing else


def test_render_empty_string_cell_is_empty_not_dash():
    # value_a="" must render an EMPTY cell (<td></td>), NOT the em-dash reserved
    # for None. Assert the positive structure AND zero dashes (a strong pin, not
    # a global "no dash anywhere" negative).
    html = str(render_config_diff_table([{"key": "a", "value_a": "", "value_b": "x"}]))
    assert "<td></td>" in html
    assert html.count("—") == 0


def test_render_large_diff_builds_without_error():
    diffs = [{"key": f"k{i:04d}", "value_a": str(i), "value_b": str(i + 1)} for i in range(300)]
    html = str(render_config_diff_table(diffs))
    assert "300 differing config keys" in html
