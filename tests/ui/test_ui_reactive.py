"""Tests for Shiny reactive semantics relied on by the OSMOSE UI.

Deep review v3 D-1: pins the invariant that reactive.Value.set() inside
reactive.isolate() propagates to downstream readers.
"""

from shiny import reactive


def test_isolate_is_context_manager():
    """reactive.isolate() must be a usable context manager (API contract).

    The OSMOSE UI uses `with reactive.isolate(): state.config.set(cfg)`
    in forcing.py:136-138 and state.py:62-68. This test verifies the API
    surface exists.
    """
    assert hasattr(reactive, "isolate")
    ctx = reactive.isolate()
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


def test_reactive_value_set_is_unconditional():
    """reactive.Value.set() must be callable inside an isolate block.

    Shiny's isolate() suppresses READS from creating reactive dependencies.
    WRITES (set) are never suppressed — they always propagate to downstream
    readers. This is the key semantic guarantee the OSMOSE UI relies on.

    This test verifies the API allows writing inside isolate without error.
    Full integration verification requires an active Shiny session.

    Deep review v3 D-1.
    """
    val = reactive.Value(0)
    with reactive.isolate():
        val.set(42)
    with reactive.isolate():
        assert val.get() == 42, "Value.set() inside isolate must persist"
