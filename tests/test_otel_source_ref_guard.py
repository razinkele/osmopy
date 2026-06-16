"""Regression test for the Shiny OTel source-ref hardening shim in app.py.

Shiny 1.6.3's ``set_renderer`` calls ``extract_source_ref`` unconditionally; it runs
``inspect.getsourcelines`` whose ``except`` clause does NOT catch ``tokenize.TokenError``.
When a long-running process's source drifts from disk (e.g. edited under it), that
uncaught error aborts ``server()`` and silently breaks every interactive handler.
``app._harden_shiny_otel_source_ref`` wraps it so the failure degrades to empty attrs.
"""

import tokenize

import app
from shiny.session import _session as _sess


def test_guard_applied_on_import():
    """Importing app installs the guard on the call site Shiny actually uses."""
    assert getattr(_sess.extract_source_ref, "_osmose_guarded", False) is True


def test_guard_swallows_tokenerror(monkeypatch):
    """A TokenError (the exact uncaught type) from extraction becomes {} , not a crash."""

    def raiser(func):
        raise tokenize.TokenError("unterminated string literal (detected at line 1)", (1, 25))

    # Replace with an un-guarded raiser, re-apply the shim, and confirm it swallows.
    monkeypatch.setattr(_sess, "extract_source_ref", raiser, raising=False)
    app._harden_shiny_otel_source_ref()
    assert _sess.extract_source_ref(lambda: None) == {}


def test_guard_preserves_normal_attrs(monkeypatch):
    """Non-failing extraction is passed through unchanged."""

    def ok(func):
        return {"code.file.path": "/x.py"}

    monkeypatch.setattr(_sess, "extract_source_ref", ok, raising=False)
    app._harden_shiny_otel_source_ref()
    assert _sess.extract_source_ref(lambda: None) == {"code.file.path": "/x.py"}
