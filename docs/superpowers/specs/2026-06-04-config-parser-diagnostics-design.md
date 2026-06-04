# Better config parser diagnostics — Design

**Date:** 2026-06-04
**Status:** Approved direction (brainstormed; codebase-grounded). Small DX feature.

## Motivation

`OsmoseConfigReader` (`osmose/config/reader.py`) silently degrades on malformed config: an
unparseable line is logged as `_log.warning("Skipping unparseable line in %s: %r", filepath.name,
line)` — **no line number**, no structured record, just a `self.skipped_lines` count. **Duplicate
keys within a file silently last-wins** with no warning (a typo'd repeat overrides invisibly). The
recursive issues (circular reference, missing sub-config, path-escape) are also log-only. A user
who mistypes a config gets a scattered log line they can't locate. This adds **structured,
line-located diagnostics** so the problem is findable — without changing what `read()` returns.

## Verified context (audit)

- `osmose/config/reader.py::OsmoseConfigReader`:
  - `read(master_file) -> dict[str,str]` resets `skipped_lines`/`key_case_map`, recurses, returns
    the flat dict (+ injects `_osmose.config.dir`).
  - `read_file(filepath) -> dict[str,str]` iterates `for line in f:` (no line number), splits on
    `SEPARATORS = re.compile(r"\s*[=;,:\t]\s*")` with `maxsplit=1`; `len(parts)==2` → store
    (lowercased key, value `.rstrip(";,:\t =")`), record case in `key_case_map`; else
    `_log.warning("Skipping unparseable line …")` + `skipped += 1`.
  - `_read_recursive` warns (log-only) on: circular ref, sub-file path escaping the config dir,
    missing referenced sub-config.
  - Keys are lowercased for storage; `key_case_map[lower] = raw_key`. **Duplicate detection must be
    on the lowercased key** (matching how the reader dedups/overwrites).
- Existing consumers of `skipped_lines`: confirm via grep (the diagnostics are additive; do not
  remove `skipped_lines`).
- `config_validation.py` is a SEPARATE concern (engine key-allowlist), not parse-level diagnostics
  — no overlap. (The in-loop review must confirm no existing parse-diagnostics/line-number surface
  to avoid duplication — the recurring "it already exists" lesson.)
- CI lints `osmose/ ui/ tests/` (NOT `scripts/`).

## Architecture

Enrich the reader to **collect** structured diagnostics during parsing; add a formatter + a CLI.
The `read()`/`read_file()` return contract is unchanged (still the flat dict) — purely additive.

### 1. `ConfigDiagnostic` dataclass (`osmose/config/reader.py`)

```python
@dataclass(frozen=True)
class ConfigDiagnostic:
    file: str          # filepath.name (or full path)
    lineno: int | None # 1-based line number; None for whole-file/recursive issues
    line: str          # the offending line text ("" when N/A)
    reason: str        # "unparseable" | "empty_key" | "duplicate_key" | "circular_ref"
                       #   | "missing_subconfig" | "path_escape"
    detail: str        # human context (e.g. the key, the sub-path)
```

**Severity classes** (used by the CLI exit code): **errors** = `{unparseable, circular_ref,
missing_subconfig, path_escape}` (the line/file is genuinely broken); **warnings** =
`{empty_key, duplicate_key}` (surfaced but not exit-failing). A module-level
`_ERROR_REASONS: frozenset[str]` encodes this.

### 2. Reader enrichment

- `OsmoseConfigReader.__init__`: add `self.diagnostics: list[ConfigDiagnostic] = []`.
- `read()`: reset `self.diagnostics = []` alongside the existing resets.
- `read_file`: iterate with `enumerate(f, 1)` for the 1-based line number.
  - Unparseable line (`len(parts) != 2`) → append `ConfigDiagnostic(file, lineno, line,
    "unparseable", "")` AND keep the log warning (now including the line number); `skipped += 1`
    unchanged.
  - **Empty key** (`len(parts) == 2` but the lowercased `key == ""`): this is the separator-led
    line class (`=value`, `,,`, `;;`, `:`). Classify on the **post-rstrip `value`** the reader
    already computes (`value = parts[1].strip().rstrip(";,:\t =")`) — e.g. `,,` splits to
    `["", ","]` and rstrips to `value == ""`. Distinguish two sub-cases:
    - `value == ""` too (a `,,`/`;;` **blank spacer row** — a real, intentional pattern in shipped
      CSVs, e.g. `data/eec_full/eec_param-output.csv:5,7,8,24`): **benign — emit NO diagnostic.**
    - `value != ""` (a genuine `=value` typo that lost its key): emit `ConfigDiagnostic(file,
      lineno, line, "empty_key", detail="missing key before separator")`.
    - In **both** sub-cases the **stored result is UNCHANGED** (today's reader stores
      `result[""] = value`; we keep that exactly — additive-only, parity-safe; we do NOT alter what
      the dict contains). And an empty key is **excluded from duplicate tracking** (so repeated
      `,,` spacer rows never produce a `duplicate_key` storm).
  - **Duplicate key within this file** (only for **non-empty** lowercased keys): track keys seen
    *in this file*; if a non-empty key recurs → append `ConfigDiagnostic(file, lineno, line,
    "duplicate_key", detail=f"overrides earlier '{raw_key}'")`. Last-wins behavior is **unchanged**
    (still `result[key] = value`); the diagnostic just makes it visible. (Cross-file overrides are
    NOT flagged — intentional in OSMOSE's recursive sub-config model.)
  - **Why this matters (review finding):** the empty-key path is a pre-existing *silent footgun* —
    a `=value` line that lost its key parses as `len==2` (NOT unparseable) and silently pollutes
    the dict. Surfacing it (without changing storage) is the highest-value part of this feature.
    The benign-blank-row carve-out is what keeps shipped configs (eec_full) diagnostic-free.
- `_read_recursive`: alongside the existing log warnings, append diagnostics for `circular_ref`,
  `missing_subconfig`, `path_escape` (lineno=None; detail = the path/key).

### 3. `format_diagnostics(diags) -> str` (module-level)

Human-readable report grouped by file. **Line format (explicit contract):** when `lineno is not
None` → `"<file>:<lineno>: <reason> — <line>"`; when `lineno is None` (recursive issues) →
`"<file>: <reason> — <detail>"` (NO stray `:None:`). Ends with a one-line summary
(`"N issue(s): X unparseable, Y empty-key, Z duplicate-key, …"`). Empty list →
`"No config issues found."`.

### 4. CLI `scripts/check_config.py`

`argparse` (mirrors `compare_runs.py` conventions): `--config <master file>` (required). Reads it
via `OsmoseConfigReader`, prints `format_diagnostics(reader.diagnostics)`, and **exits `1` only
when an ERROR-class diagnostic is present** (`unparseable`/`circular_ref`/`missing_subconfig`/
`path_escape` — i.e. `any(d.reason in _ERROR_REASONS)`); `empty_key`/`duplicate_key` warnings print
but exit `0`. This keeps the CLI from red-failing a valid shipped config. (No `--json` for v1 —
YAGNI; the structured list is available in-process for a future UI/JSON consumer.) The CLI targets
**config masters**, not arbitrary data/map CSVs — `read()` only recurses `osmose.configuration.*`
references, so map/mask matrix CSVs are never parsed as config (pointing the CLI directly at one is
out of the promised scope).

## Data flow

`read(master)` → recursive parse, appending `ConfigDiagnostic`s as issues arise → returns the
unchanged flat dict; `reader.diagnostics` holds the structured issues; CLI formats + surfaces them.

## Error handling / edge cases

- A value that legitimately contains a separator is fine — `maxsplit=1` keeps it in the value;
  only lines that produce `len(parts) != 2` (i.e. no separator at all) are "unparseable".
- `osmose.configuration.*` keys may appear multiple times legitimately (multiple sub-config
  references) — these are DISTINCT keys, not duplicates, so duplicate detection (on the full
  lowercased key) won't false-positive on them unless the *same* key literally repeats.
- Empty/comment lines are skipped before the split (unchanged) → never diagnosed.
- Whole-file failures (too-large → existing `ValueError`; unreadable) keep current behavior.
- Backward-compat: valid configs produce an empty `diagnostics` list and a byte-identical dict.

## Testing (`tests/test_config_reader_diagnostics.py`)

- Unparseable line at a known position → one `unparseable` diagnostic with the correct `lineno`
  and `line` (write a temp config: a couple valid lines, one junk line, assert lineno).
- Duplicate key within a file → one `duplicate_key` diagnostic on the second occurrence; the dict
  still holds the last value (behavior unchanged).
- Missing sub-config / path-escape / circular ref → the corresponding diagnostic (lineno None,
  detail set).
- Empty-key cases: `=value` (non-empty value) → one `empty_key` diagnostic; `,,`/`;;` (empty
  value) → **NO** diagnostic (benign blank row); repeated `,,` rows → still no diagnostic and **no**
  `duplicate_key` (empty keys excluded from duplicate tracking). In all cases the returned dict is
  unchanged from today.
- Valid config → `diagnostics == []` AND the returned dict equals the pre-change result
  (regression-lock the additive-only guarantee).
- **ALL five shipped masters** (baltic, baltic_ev, eec, eec_full, minimal) read with `diagnostics
  == []` (parametrized) — guards against false positives on real configs. (Review confirmed
  eec_full's `,,` spacer rows must NOT produce diagnostics; baltic et al. are clean.)
- `format_diagnostics` on a mixed list (incl. a `lineno=None` recursive diagnostic → asserts no
  `:None:` in the output) + on `[]`.
- CLI smoke: a malformed temp config → exit 1 + report; a clean one → exit 0. (CLI in `scripts/`,
  lint-exempt; the reader + formatter + tests are in linted dirs and must be ruff-clean.)

## Scope / YAGNI

- **In:** line numbers, the `ConfigDiagnostic` structured list (unparseable + empty-key +
  within-file duplicate-key + the three recursive issues), severity classes for the CLI exit code,
  `format_diagnostics`, the CLI, tests.
- **Out:** column-level positions (not meaningful for `key=value` lines — line number is the
  locator); cross-file override "duplicates" (intentional); auto-fixing; UI surfacing (a thin
  follow-on now that diagnostics are structured); `--json` CLI output; any change to `read()`'s
  return type, `key_case_map`, or `skipped_lines`.

## Honest limitations

- Reports, does not auto-fix.
- "Column" is intentionally omitted (low value for this format).
- Duplicate detection is **within a single file** only (cross-file override is a valid OSMOSE
  pattern and would be noise).

## Delivery

Single small PR: `osmose/config/reader.py` (dataclass + enrichment + `format_diagnostics`),
`scripts/check_config.py`, `tests/test_config_reader_diagnostics.py`, a docs/CHANGELOG note. No
engine changes, no behavior change to the returned config dict.
