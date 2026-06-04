# Config parser diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `OsmoseConfigReader` structured, line-located parse diagnostics (unparseable lines, empty keys, within-file duplicate keys, recursive-reference issues) + a formatter + a CLI — without changing what `read()` returns.

**Architecture:** Additive enrichment of `osmose/config/reader.py`: a frozen `ConfigDiagnostic` dataclass, a `self.diagnostics` list populated during parsing (with 1-based line numbers), module-level `format_diagnostics` + `diagnostics_have_errors` helpers, and a `scripts/check_config.py` CLI. The returned flat dict, `key_case_map`, and `skipped_lines` are byte-identical to today (verified against all 5 shipped configs in review).

**Tech Stack:** Python 3.12 (dataclasses, re), pytest, ruff. Tests: `.venv/bin/python -m pytest`. Scripts run with `PYTHONPATH=/home/razinka/osmose/osmose-python`.

**Reference spec:** `docs/superpowers/specs/2026-06-04-config-parser-diagnostics-design.md` (reviewed clean over 2 in-loop rounds; all 5 shipped masters verified diagnostic-free).

---

## Verified facts (audit — use exactly)

- `osmose/config/reader.py` current shape (confirmed):
  - imports: `from __future__ import annotations`, `import re`, `from pathlib import Path`,
    `from osmose.logging import setup_logging`; `_log = setup_logging("osmose.config")`.
  - `class OsmoseConfigReader`: `SEPARATORS = re.compile(r"\s*[=;,:\t]\s*")`,
    `COMMENT_CHARS = {"#", "!"}`. `__init__` sets `self.key_case_map = {}` + `self.skipped_lines = 0`.
  - `read(master_file)` resets `self.skipped_lines = 0` and `self.key_case_map = {}`, calls
    `_read_recursive`, injects `flat["_osmose.config.dir"]`, returns the dict.
  - `_read_recursive` log-warns (no structured record) on: circular ref (`resolved in _seen`),
    path-escape (`not resolved_sub.is_relative_to(config_dir)`), missing sub-config
    (`else` of `sub_path.exists()`).
  - `read_file` loops `for line in f:` (NO line number); `parts = self.SEPARATORS.split(line,
    maxsplit=1)`; `len(parts)==2` → `raw_key=parts[0].strip()`, `key=raw_key.lower()`,
    `value=parts[1].strip().rstrip(";,:\t =")`, `result[key]=value`, `key_case_map[key]=raw_key`;
    else `_log.warning("Skipping unparseable line in %s: %r", filepath.name, line)` + `skipped+=1`.
- A separator-led line splits to `len==2` with `key==""`: `,,`/`;;`/`:`/`=` → post-rstrip
  `value==""` (benign blank row); `=value` → `value!=""` (real error). The reader stores
  `result[""]=value` today — **keep that unchanged**.
- Duplicate detection is on the **lowercased** key, **within a single file**, **non-empty keys
  only** (empty keys excluded). All 5 shipped masters (baltic, baltic_ev, eec, eec_full, minimal)
  produce ZERO diagnostics under these rules (eec_full has `,,` spacer rows that must stay benign).
- `skipped_lines` is read only by `tests/test_config_reader_errors.py` — keep it unchanged.
- CI lints `osmose/ ui/ tests/` (NOT `scripts/`).

## File Structure

- Modify: `osmose/config/reader.py` — `ConfigDiagnostic` dataclass, `_ERROR_REASONS`,
  `format_diagnostics`, `diagnostics_have_errors`, `self.diagnostics` init/reset, `read_file` +
  `_read_recursive` enrichment.
- Create: `scripts/check_config.py` — CLI.
- Create: `tests/test_config_reader_diagnostics.py` — unit + shipped-config regression + CLI smoke.
- Modify: `CHANGELOG.md` — Added note.

> **TEST-FILE IMPORT CONVENTION (ruff E402):** keep ALL module-level imports of
> `tests/test_config_reader_diagnostics.py` in the top block; each task edits that block to add
> names and appends only test functions. Never append imports after functions.
>
> **RUFF FORMAT-FIRST (avoids the "green check, red format" CI trap):** the code blocks below are
> NOT pre-wrapped to ruff's style — the multi-line `ConfigDiagnostic(...)` calls will be re-wrapped
> by `ruff format`. So in every task's verify step, **run `.venv/bin/ruff format <touched files>`
> FIRST, then `ruff check` + `ruff format --check`** (CI runs BOTH `ruff check` AND `ruff format
> --check` on `osmose/ ui/ tests/`; `ruff check` passing alone is not enough). Re-run the task's
> tests after formatting.

---

## Task 1: `ConfigDiagnostic` + formatter + reader wiring (no loop change yet)

**Files:**
- Modify: `osmose/config/reader.py`
- Test: `tests/test_config_reader_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config_reader_diagnostics.py`:
```python
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
    assert ":None:" not in out  # None lineno must not leak
    assert "2 issue(s):" in out


def test_diagnostics_have_errors():
    err = [ConfigDiagnostic("a", 1, "x", "unparseable", "")]
    warn = [ConfigDiagnostic("a", 1, "x", "duplicate_key", "d")]
    assert diagnostics_have_errors(err) is True
    assert diagnostics_have_errors(warn) is False
    assert diagnostics_have_errors([]) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -q`
Expected: FAIL (`ImportError: cannot import name 'ConfigDiagnostic'`).

- [ ] **Step 3: Implement in `osmose/config/reader.py`**

Extend the top imports — change `from __future__ import annotations` block to also import the
dataclass decorator. After the existing imports, the import block becomes:
```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.config")
```
Then add, immediately after the `_log = ...` line (module level, before `class OsmoseConfigReader`):
```python
@dataclass(frozen=True)
class ConfigDiagnostic:
    """A structured, line-located config-parse issue."""

    file: str
    lineno: int | None
    line: str
    reason: str  # unparseable|empty_key|duplicate_key|circular_ref|missing_subconfig|path_escape
    detail: str


_ERROR_REASONS: frozenset[str] = frozenset(
    {"unparseable", "circular_ref", "missing_subconfig", "path_escape"}
)


def diagnostics_have_errors(diagnostics: list[ConfigDiagnostic]) -> bool:
    """True if any diagnostic is ERROR-class (vs the empty_key/duplicate_key warnings)."""
    return any(d.reason in _ERROR_REASONS for d in diagnostics)


def format_diagnostics(diagnostics: list[ConfigDiagnostic]) -> str:
    """Human-readable report grouped by file; one line per diagnostic + a summary."""
    if not diagnostics:
        return "No config issues found."
    out: list[str] = []
    for d in diagnostics:
        if d.lineno is not None:
            body = f"{d.file}:{d.lineno}: {d.reason}"
            if d.line:
                body += f" — {d.line}"
        else:
            body = f"{d.file}: {d.reason}"
            if d.detail:
                body += f" — {d.detail}"
        out.append(body)
    counts: dict[str, int] = {}
    for d in diagnostics:
        counts[d.reason] = counts.get(d.reason, 0) + 1
    summary = ", ".join(f"{n} {r}" for r, n in sorted(counts.items()))
    out.append(f"{len(diagnostics)} issue(s): {summary}")
    return "\n".join(out)
```
Then wire the list into the reader. In `__init__`, after `self.skipped_lines: int = 0`, add:
```python
        self.diagnostics: list[ConfigDiagnostic] = []
```
In `read()`, after `self.key_case_map = {}`, add:
```python
        self.diagnostics = []
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -q` → 3 pass.
Run: `.venv/bin/python -c "import osmose.config.reader"` → ok.
Run: `.venv/bin/ruff check osmose/config/reader.py tests/test_config_reader_diagnostics.py && .venv/bin/ruff format --check osmose/config/reader.py tests/test_config_reader_diagnostics.py` (clean; if format flags, run `.venv/bin/ruff format <file>` + re-test).

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/config/reader.py tests/test_config_reader_diagnostics.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(config): ConfigDiagnostic dataclass + format/has-errors helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `read_file` enrichment (line numbers, unparseable, empty_key, duplicate_key)

**Files:**
- Modify: `osmose/config/reader.py`
- Test: `tests/test_config_reader_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Edit the TOP import block of `tests/test_config_reader_diagnostics.py` to add `OsmoseConfigReader`:
```python
from osmose.config.reader import (
    ConfigDiagnostic,
    OsmoseConfigReader,
    diagnostics_have_errors,
    format_diagnostics,
)
```
Append these test functions to the END of the file:
```python
def _write(tmp_path, text):
    p = tmp_path / "cfg.csv"
    p.write_text(text)
    return p


def test_unparseable_line_has_lineno(tmp_path):
    # line 1 valid, line 2 comment, line 3 junk (no separator)
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
    assert out[""] == "orphanvalue"  # storage unchanged (additive-only)


def test_blank_spacer_rows_are_benign(tmp_path):
    # ",," / ";;" spacer rows: empty key AND empty value -> no diagnostic, no duplicate
    p = _write(tmp_path, "a;1\n,,\n,,\n;;\n")
    r = OsmoseConfigReader()
    r.read_file(p)
    assert r.diagnostics == []  # benign


def test_duplicate_key_within_file(tmp_path):
    p = _write(tmp_path, "Foo;1\nbar;2\nfoo;3\n")  # foo repeats (case-insensitive)
    r = OsmoseConfigReader()
    out = r.read_file(p)
    diags = [d for d in r.diagnostics if d.reason == "duplicate_key"]
    assert len(diags) == 1 and diags[0].lineno == 3
    assert out["foo"] == "3"  # last-wins unchanged


def test_additive_only_dict_unchanged(tmp_path):
    # The returned dict must equal what the reader produced before diagnostics existed.
    p = _write(tmp_path, "a;1\nb;2\n,,\nc;3\n")
    r = OsmoseConfigReader()
    out = r.read_file(p)
    assert out == {"a": "1", "b": "2", "": "", "c": "3"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -k "unparseable_line_has_lineno or empty_key or spacer or duplicate_key or additive_only" -q`
Expected: FAIL (current `read_file` has no line numbers / no empty_key / no duplicate diagnostics).

- [ ] **Step 3: Replace `read_file`'s parse loop**

In `osmose/config/reader.py`, replace the body of `read_file` from `result: dict[str, str] = {}`
through `return result` with this (keep the `st = filepath.stat()` size guard above it unchanged):
```python
        result: dict[str, str] = {}
        skipped = 0
        seen_keys: set[str] = set()
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for lineno, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line[0] in self.COMMENT_CHARS:
                    continue
                parts = self.SEPARATORS.split(line, maxsplit=1)
                if len(parts) == 2:
                    raw_key = parts[0].strip()
                    key = raw_key.lower()
                    value = parts[1].strip()
                    # Strip trailing separators (e.g., "true," → "true")
                    value = value.rstrip(";,:\t =")
                    if key == "":
                        # Separator-led line. ",,"/";;" (empty value) are benign blank rows;
                        # "=value" (lost its key) is a real error. Storage is unchanged below;
                        # empty keys are never tracked for duplicates.
                        if value != "":
                            self.diagnostics.append(
                                ConfigDiagnostic(
                                    filepath.name, lineno, line, "empty_key",
                                    "missing key before separator",
                                )
                            )
                    elif key in seen_keys:
                        self.diagnostics.append(
                            ConfigDiagnostic(
                                filepath.name, lineno, line, "duplicate_key",
                                f"overrides earlier '{self.key_case_map.get(key, key)}'",
                            )
                        )
                    result[key] = value
                    self.key_case_map[key] = raw_key
                    if key != "":
                        seen_keys.add(key)
                else:
                    self.diagnostics.append(
                        ConfigDiagnostic(filepath.name, lineno, line, "unparseable", "")
                    )
                    _log.warning(
                        "Skipping unparseable line %d in %s: %r", lineno, filepath.name, line
                    )
                    skipped += 1
        self.skipped_lines += skipped
        return result
```
Key points: `enumerate(f, 1)` for physical line numbers; the duplicate detail reads
`key_case_map.get(key)` BEFORE the overwrite (so it names the earlier raw key); empty keys are
never added to `seen_keys`; `result[key]=value` and `key_case_map[key]=raw_key` happen for every
`len==2` line exactly as before (additive-only).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/test_config_reader_errors.py -q` → still green (the
`skipped_lines` behavior is unchanged).
Run: `.venv/bin/ruff check osmose/config/reader.py tests/test_config_reader_diagnostics.py && .venv/bin/ruff format --check osmose/config/reader.py tests/test_config_reader_diagnostics.py`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/config/reader.py tests/test_config_reader_diagnostics.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(config): line-numbered unparseable/empty_key/duplicate_key diagnostics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `_read_recursive` enrichment + shipped-config regression

**Files:**
- Modify: `osmose/config/reader.py`
- Test: `tests/test_config_reader_diagnostics.py`

- [ ] **Step 1: Write failing tests**

First, edit the TOP import block to add `import pytest` and `from pathlib import Path` (keep all
imports at the top — E402). The block becomes:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import (
    ConfigDiagnostic,
    OsmoseConfigReader,
    diagnostics_have_errors,
    format_diagnostics,
)
```
Then **first** confirm the real shipped-master filenames: `ls data/*/*_all-parameters.csv` (the
list below uses the expected names; substitute the actual ones — the test `skip`s any that are
absent, so it stays green regardless). Append these test functions to the END of the file:
```python
def test_missing_subconfig_diagnostic(tmp_path):
    master = tmp_path / "master.csv"
    master.write_text("osmose.configuration.sub;does_not_exist.csv\n")
    r = OsmoseConfigReader()
    r.read(master)
    diags = [d for d in r.diagnostics if d.reason == "missing_subconfig"]
    assert len(diags) == 1 and diags[0].lineno is None
    assert "does_not_exist.csv" in diags[0].detail


@pytest.mark.parametrize(
    "master",
    [
        "data/baltic/baltic_all-parameters.csv",
        "data/baltic_ev/baltic_ev_all-parameters.csv",
        "data/eec/osm_all-parameters.csv",
        "data/eec_full/eec_all-parameters.csv",
        "data/minimal/osm_all-parameters.csv",
    ],
)
def test_shipped_masters_have_no_diagnostics(master):
    p = Path(master)
    if not p.is_file():
        pytest.skip(f"shipped master not present: {master}")
    r = OsmoseConfigReader()
    r.read(p)
    assert r.diagnostics == [], format_diagnostics(r.diagnostics)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -k "missing_subconfig or shipped_masters" -v`
Expected: `missing_subconfig` FAILS (no diagnostic emitted yet); `shipped_masters` should already
PASS (read_file enrichment from T2 already keeps them clean) — confirm it does.

- [ ] **Step 3: Implement `_read_recursive` diagnostics**

In `_read_recursive`, alongside each existing `_log.warning`, append a diagnostic. Replace the
three warning sites:

Circular ref (`if resolved in _seen:`):
```python
        if resolved in _seen:
            _log.warning("Circular config reference skipped: %s", filepath)
            self.diagnostics.append(
                ConfigDiagnostic(filepath.name, None, "", "circular_ref", str(filepath))
            )
            return
```
Path-escape:
```python
                if not resolved_sub.is_relative_to(config_dir):
                    _log.warning(
                        "Sub-file path escapes config directory, skipping: %s (from key %s)",
                        sub_path,
                        key,
                    )
                    self.diagnostics.append(
                        ConfigDiagnostic(
                            filepath.name, None, "", "path_escape", f"{sub_path} (from key {key})"
                        )
                    )
                    continue
```
Missing sub-config:
```python
                else:
                    _log.warning("Referenced sub-config not found: %s (from key %s)", sub_path, key)
                    self.diagnostics.append(
                        ConfigDiagnostic(
                            filepath.name, None, "", "missing_subconfig",
                            f"{sub_path} (from key {key})",
                        )
                    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -v` → all pass (incl. the
5 shipped-master cases — any absent master is skipped, present ones assert `[]`).
Run: `.venv/bin/ruff check osmose/config/reader.py tests/test_config_reader_diagnostics.py && .venv/bin/ruff format --check osmose/config/reader.py tests/test_config_reader_diagnostics.py`.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/config/reader.py tests/test_config_reader_diagnostics.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(config): recursive-ref diagnostics + shipped-config regression

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: CLI `scripts/check_config.py`

**Files:**
- Create: `scripts/check_config.py`
- Test: `tests/test_config_reader_diagnostics.py` (CLI smoke via subprocess-free `main()` call)

- [ ] **Step 1: Write failing test**

`scripts/` is not an importable package, so the test loads the CLI module by path via `importlib`
(no top-import change needed). Append this test to the END of the file:
```python
def test_cli_exit_codes(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "check_config", Path("scripts/check_config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # unparseable line -> ERROR-class -> exit 1
    bad = tmp_path / "bad.csv"
    bad.write_text("good;1\njunkline\n")
    assert mod.main(["--config", str(bad)]) == 1

    # only a duplicate_key (warning) -> exit 0
    warn = tmp_path / "warn.csv"
    warn.write_text("a;1\na;2\n")
    assert mod.main(["--config", str(warn)]) == 0

    # clean -> exit 0
    good = tmp_path / "good.csv"
    good.write_text("a;1\nb;2\n")
    assert mod.main(["--config", str(good)]) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -k cli_exit_codes -q`
Expected: FAIL (`scripts/check_config.py` does not exist).

- [ ] **Step 3: Create `scripts/check_config.py`**

```python
#!/usr/bin/env python3
"""Report structured parse diagnostics for an OSMOSE config master file.

Reads the config (recursively) and prints line-located issues: unparseable lines,
empty keys, within-file duplicate keys, and recursive-reference problems (circular /
missing sub-config / path-escape). Exits 1 only when an ERROR-class issue is present
(unparseable / circular_ref / missing_subconfig / path_escape); empty-key and
duplicate-key warnings print but exit 0. For config MASTER files, not data/map CSVs.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/check_config.py --config <master.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.config.is_file():
        p.error(f"config file not found: {args.config}")

    from osmose.config.reader import (
        OsmoseConfigReader,
        diagnostics_have_errors,
        format_diagnostics,
    )

    reader = OsmoseConfigReader()
    reader.read(args.config)
    print(format_diagnostics(reader.diagnostics))
    return 1 if diagnostics_have_errors(reader.diagnostics) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass + smoke a real config**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py -q` → all pass.
Run (real clean config → exit 0, prints "No config issues found."):
`PYTHONPATH=/home/razinka/osmose/osmose-python .venv/bin/python scripts/check_config.py --config data/baltic/baltic_all-parameters.csv` (adjust to the real master name from `ls data/baltic/*_all-parameters.csv`). Expected: "No config issues found." and exit 0.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add scripts/check_config.py tests/test_config_reader_diagnostics.py
git -C /home/razinka/osmose/osmose-python commit -m "feat(config): check_config.py CLI for parse diagnostics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Docs + full verification + lint

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG note**

Under `## [Unreleased]` → `### Added` (create the subsection if absent, Keep-a-Changelog order:
Added before Fixed), add:
```markdown
- **config (parser diagnostics):** `OsmoseConfigReader` now collects structured, line-located
  parse issues in `reader.diagnostics` — unparseable lines, empty keys (e.g. a `=value` that lost
  its key), within-file duplicate keys, and recursive-reference problems (circular / missing
  sub-config / path-escape) — plus `format_diagnostics()` and a `scripts/check_config.py` CLI.
  Additive only: the parsed config dict is unchanged.
```

- [ ] **Step 2: Full verification**

Run: `.venv/bin/python -m pytest tests/test_config_reader_diagnostics.py tests/test_config_reader_errors.py -v` (report counts; all pass).
Run: `.venv/bin/python -m pytest tests/ -k "config" -q` (report pass/fail; classify any failure pre-existing vs caused — if unsure, say so).
Run: `.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/` (clean on touched files; if format flags a file YOU touched, run `.venv/bin/ruff format <file>` and re-test; untouched flags → leave + note).

- [ ] **Step 3: Commit + finish**

```bash
git -C /home/razinka/osmose/osmose-python add CHANGELOG.md
git -C /home/razinka/osmose/osmose-python commit -m "docs(changelog): config parser diagnostics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then use superpowers:requesting-code-review then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:** `ConfigDiagnostic` + 6 reasons → T1 (dataclass) + T2/T3 (emitters); line numbers
via `enumerate(f,1)` → T2; empty_key (value!="" only) + benign `,,` + empty-keys-excluded-from-dups
→ T2; within-file duplicate (non-empty, lowercased) → T2; recursive diagnostics → T3;
`format_diagnostics` None-contract → T1; severity/`_ERROR_REASONS` + CLI exit codes → T1 +
T4; additive-only dict-equality → T2 test; all-5-shipped-masters diagnostic-free → T3 test; CLI for
masters not data CSVs → T4 docstring; docs → T5; out-of-scope (column positions, cross-file dups,
auto-fix, UI, `--json`, dict/`key_case_map`/`skipped_lines` changes) → not in plan, per spec. ✅

**Placeholder scan:** no TBD/TODO; every step has exact before/after code + commands. The two
"adjust to the real master filename" notes (T3/T4) are grounded instructions with the `ls` command
to resolve them + a `skip` fallback in the test, not placeholders. ✅

**Type consistency:** `ConfigDiagnostic(file, lineno, line, reason, detail)` constructed
positionally the same way in T2 (3 sites), T3 (3 sites), T1 tests; `diagnostics_have_errors` /
`format_diagnostics` / `_ERROR_REASONS` defined in T1, used in T4 CLI; `self.diagnostics` init (T1
`__init__`) + reset (T1 `read`) + appended (T2/T3). The reason strings are the exact 6 from the
dataclass comment + `_ERROR_REASONS`. ✅
