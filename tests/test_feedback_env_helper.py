"""Default-run coverage of the e2e store-isolation helper (`tests/_e2e_feedback_env.py`).

The helper itself only ever runs under `-m e2e`, which CI does not run, so a regression in it
would sit undetected until someone ran the browser tests by hand. These tests cost nothing and
pin the two properties that actually matter:

* it sets **both** store variables. The contacts one was simply missing before, so the first e2e
  submission carrying an email address wrote a real reporter address into the developer's own
  ``data/feedback/contacts.jsonl`` — gitignored, so it would never have appeared in a diff.
* it **restores** the environment at teardown. The previous import-time ``os.environ[...] = ...``
  never did: `pytest tests/` with no `-m e2e` still *imports* the e2e modules (the marker filters
  execution, not collection), so the variable leaked into every other test in the session.

The fixture is driven as a generator through ``__wrapped__`` rather than through a pytest
sub-run: setup is everything before the ``yield`` and teardown everything after, which is exactly
the lifecycle under test, and it keeps these tests free of subprocess and ``sys.path`` plumbing.
"""

from __future__ import annotations

import os

import pytest

from tests._e2e_feedback_env import feedback_env_fixture, store_fixture

_BOTH = ("OSMOSE_FEEDBACK_FILE", "OSMOSE_CONTACTS_FILE")


def _drive(stem: str):
    """Yield ``(paths, env_during)`` for one setup/teardown cycle of the module fixture."""
    gen = feedback_env_fixture(stem).__wrapped__()
    paths = next(gen)
    during = {k: os.environ.get(k) for k in _BOTH}
    try:
        next(gen)
    except StopIteration:
        pass
    return paths, during


def test_helper_sets_both_store_variables():
    """Both variables set, to this module's own dedicated pair, and never to the same file."""
    (feedback, contacts), during = _drive("unit_probe")

    assert during["OSMOSE_FEEDBACK_FILE"] == str(feedback), (
        f"OSMOSE_FEEDBACK_FILE was {during['OSMOSE_FEEDBACK_FILE']!r}, want {str(feedback)!r}"
    )
    assert during["OSMOSE_CONTACTS_FILE"] == str(contacts), (
        f"OSMOSE_CONTACTS_FILE was {during['OSMOSE_CONTACTS_FILE']!r}, want {str(contacts)!r} — "
        "an unset contacts var sends real reporter addresses to the default store"
    )
    assert feedback != contacts, "both stores resolve to ONE file; records and PII would mix"
    assert "unit_probe" in feedback.name, f"not a per-module file: {feedback.name!r}"


def test_helper_restores_the_environment_at_teardown():
    """Neither variable may survive teardown — the property the import-time version lacked."""
    _, during = _drive("unit_probe")
    assert all(during[k] for k in _BOTH), "fixture did not set the variables at all"

    leaked = {k: os.environ.get(k) for k in _BOTH if os.environ.get(k) is not None}
    assert not leaked, f"e2e store variables survived teardown: {leaked}"


def test_each_module_gets_its_own_files():
    """Two stems must never resolve to the same paths — the old shared-file race."""
    (fa, ca), _ = _drive("alpha")
    (fb, cb), _ = _drive("beta")
    assert {fa, ca}.isdisjoint({fb, cb}), (
        f"two modules share store files: {sorted(map(str, {fa, ca} & {fb, cb}))}"
    )


def test_collecting_the_e2e_modules_does_not_set_the_variables():
    """Positive control, asserted about THIS process.

    Importing the e2e modules is what pytest does at collection whenever they are not ignored.
    If either still mutated the environment at import, this fails — and this is the assertion
    that would have caught the original defect, since the e2e tests never run in CI.
    """
    pytest.importorskip("playwright", reason="e2e modules are not collectable without playwright")
    import tests.test_e2e_feedback
    import tests.test_e2e_feedback_modal  # noqa: F401

    leaked = {k: os.environ.get(k) for k in _BOTH if os.environ.get(k) is not None}
    assert not leaked, (
        f"importing the e2e modules set {sorted(leaked)} — every test in this session now reads "
        "a store it did not choose"
    )


def test_store_fixture_is_function_scoped():
    """Per-test cleanup, not per-module: one test's records must not be visible to the next.

    ``feedback_env`` is module-scoped (the app subprocess outlives each test) while the store
    wiping must happen per test. Getting these two the same would let a stale record satisfy a
    later assertion.
    """
    assert store_fixture()._fixture_function_marker.scope == "function"
    assert feedback_env_fixture("x")._fixture_function_marker.scope == "module"
