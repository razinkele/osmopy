from __future__ import annotations

from osmose.config.reader import (
    ConfigDiagnostic,
    diagnostics_have_errors,
    format_diagnostics,
)


def test_format_diagnostics_empty():
    assert format_diagnostics([]) == "No config issues found."


def test_format_diagnostics_lineno_and_none():
    diags = [
        ConfigDiagnostic("a.csv", 5, "junk line", "unparseable", ""),
        ConfigDiagnostic("a.csv", None, "", "missing_subconfig", "sub.csv (from key x)"),
    ]
    out = format_diagnostics(diags)
    assert "a.csv:5: unparseable — junk line" in out
    assert "a.csv: missing_subconfig — sub.csv (from key x)" in out
    assert ":None:" not in out
    assert "2 issue(s):" in out


def test_diagnostics_have_errors():
    err = [ConfigDiagnostic("a", 1, "x", "unparseable", "")]
    warn = [ConfigDiagnostic("a", 1, "x", "duplicate_key", "d")]
    assert diagnostics_have_errors(err) is True
    assert diagnostics_have_errors(warn) is False
    assert diagnostics_have_errors([]) is False
