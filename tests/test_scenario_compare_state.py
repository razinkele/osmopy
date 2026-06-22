"""Unit tests for the pure compare-state resolver used by the Scenarios modal."""

from ui.pages.scenarios import _resolve_compare_state


class _PD:
    """Stand-in for osmose.scenarios.ParamDiff (key/value_a/value_b)."""

    def __init__(self, key, a, b):
        self.key, self.value_a, self.value_b = key, a, b


def test_none_when_either_unselected():
    assert _resolve_compare_state("", "y", lambda a, b: [])[0] == "none"
    assert _resolve_compare_state("x", "", lambda a, b: [])[0] == "none"


def test_same_when_equal():
    assert _resolve_compare_state("x", "x", lambda a, b: [_PD("k", "1", "2")]) == ("same", None)


def test_identical_when_empty_diff():
    assert _resolve_compare_state("x", "y", lambda a, b: []) == ("identical", None)


def test_error_when_compare_raises():
    def boom(a, b):
        raise FileNotFoundError("deleted")

    assert _resolve_compare_state("x", "y", boom) == ("error", None)


def test_diffs_adapter_shape():
    tag, rows = _resolve_compare_state("x", "y", lambda a, b: [_PD("k", "1", "2")])
    assert tag == "diffs"
    assert rows == [{"key": "k", "value_a": "1", "value_b": "2"}]


def test_diffs_adapter_passes_none_through():
    # An added/removed key has a None side; the adapter must preserve it so
    # classify_config_diffs can later tag it added/removed (not coerce to "").
    tag, rows = _resolve_compare_state("x", "y", lambda a, b: [_PD("k", None, "5")])
    assert tag == "diffs"
    assert rows == [{"key": "k", "value_a": None, "value_b": "5"}]


def test_success_then_same_yields_no_stale_table():
    # The resolver is stateless, so a real diff followed by an a==b selection
    # returns ("same", None) with no leftover list — this is what keeps the
    # modal from flashing a stale table.
    diffs = [_PD("k", "1", "2")]
    assert _resolve_compare_state("x", "y", lambda a, b: diffs)[0] == "diffs"
    assert _resolve_compare_state("x", "x", lambda a, b: diffs) == ("same", None)
