from __future__ import annotations

from osmose.config.reader import (
    ConfigDiagnostic,
    OsmoseConfigReader,
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


def _write(tmp_path, text):
    p = tmp_path / "cfg.csv"
    p.write_text(text)
    return p


def test_unparseable_line_has_lineno(tmp_path):
    p = _write(tmp_path, "good.key;1\n# comment\njunkline\n")
    r = OsmoseConfigReader()
    r.read_file(p)
    diags = [d for d in r.diagnostics if d.reason == "unparseable"]
    assert len(diags) == 1
    assert diags[0].lineno == 3
    assert diags[0].line == "junkline"


def test_empty_key_value_present_is_flagged(tmp_path):
    p = _write(tmp_path, "=orphanvalue\n")
    r = OsmoseConfigReader()
    out = r.read_file(p)
    diags = [d for d in r.diagnostics if d.reason == "empty_key"]
    assert len(diags) == 1 and diags[0].lineno == 1
    assert out[""] == "orphanvalue"


def test_blank_spacer_rows_are_benign(tmp_path):
    p = _write(tmp_path, "a;1\n,,\n,,\n;;\n")
    r = OsmoseConfigReader()
    r.read_file(p)
    assert r.diagnostics == []


def test_duplicate_key_within_file(tmp_path):
    p = _write(tmp_path, "Foo;1\nbar;2\nfoo;3\n")
    r = OsmoseConfigReader()
    out = r.read_file(p)
    diags = [d for d in r.diagnostics if d.reason == "duplicate_key"]
    assert len(diags) == 1 and diags[0].lineno == 3
    assert out["foo"] == "3"


def test_additive_only_dict_unchanged(tmp_path):
    p = _write(tmp_path, "a;1\nb;2\n,,\nc;3\n")
    r = OsmoseConfigReader()
    out = r.read_file(p)
    assert out == {"a": "1", "b": "2", "": "", "c": "3"}
