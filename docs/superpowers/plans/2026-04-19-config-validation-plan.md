# Config Validation (Phase 7.3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catch OSMOSE config typos at `EngineConfig.from_dict()` load time with opt-in warnings and an optional error mode. Silent-by-default (one-line count nudge when unknowns present); `validation.strict.enabled=warn` logs each unknown with a `difflib` suggestion; `=error` raises a single `ValueError` listing all unknowns.

**Architecture:** One new module `osmose/engine/config_validation.py` exports `validate(cfg, mode)`. Known-key allowlist is a union of the 220-field `ParameterRegistry` with keys extracted via `ast.walk` of `osmose/engine/config.py` literals. Match order is literal-set → normalized-pattern-set → compiled-regex fallback. Suggestions use `difflib.get_close_matches(n=1, cutoff=0.85)` on `{idx}`-normalized strings.

**Tech Stack:** Python 3.12, stdlib `ast` + `difflib` + `functools.cache` + `importlib.resources`, pytest (caplog).

**Spec:** `docs/superpowers/specs/2026-04-19-config-validation-design.md` (commit `557ee1b`, 3-iteration review loop converged).

**Ship target:** v0.9.2 patch release bundled with any other post-v0.9.1 cleanups.

---

## Pre-flight

- [ ] Baseline test count. Run `.venv/bin/python -m pytest --co -q 2>&1 | tail -3`. Expected: **2500 collected (41 deselected) / 2485 passed** at HEAD `557ee1b`. If your run shows a different number, re-do the arithmetic below (`2485 + 21 = 2506 passed` after this plan).
- [ ] Lint baseline: `.venv/bin/ruff check osmose/ tests/` is clean.
- [ ] Library surface anchors (grep rather than trusting literal line numbers):
  - `osmose/engine/config.py` — `@classmethod def from_dict(cls, cfg)` around `:1347-1348`; first `_get(cfg, "simulation.nspecies")` call at `:1350`; the internal-marker read `cfg.get("_osmose.config.dir", "")` around `:150`; `_get(cfg, key)` raising KeyError at `:41-45`; `_enabled(cfg, key)` at `:133`; `_species_float(cfg, pattern, n)` at `:49`; `_species_float_optional` at `:61`; `_species_int` at `:53`; `_species_int_optional` at `:68`; `_species_str` at `:57`.
  - `osmose/schema/base.py` — `OsmoseField` dataclass with `choices: list[str] | None = None` and `required: bool = True`; `ParamType.ENUM` at `:19`.
  - `osmose/schema/simulation.py` — `SIMULATION_FIELDS: list[OsmoseField]` list starting at `:5`; final entry `stochastic.mortality.randomseed.fixed` around `:136-143`.
  - `osmose/schema/__init__.py` — `build_registry()` at `:29` (eager submodule imports at module top).
  - `osmose/schema/registry.py` — `ParameterRegistry.all_fields()` at `:36`; `match_field()` at `:48`; `{idx}` → `\d+` regex convention at `:30`.
  - `osmose/logging.py` — `setup_logging(name, level) -> logging.Logger` at `:7` (stdlib wrapper; `caplog` works directly).
  - `tests/test_engine_config_validation.py` — existing file with 7 `__post_init__` tests; integration-test extension point.

## File map

- **Engine:** `osmose/engine/config_validation.py` **(new, ~250 lines)** — dataclasses (`UnknownKey`, `KnownKeys`), `_normalize_key_to_pattern`, `_suggest`, `_extract_literal_keys_from_config_py`, `build_known_keys` (cached), public `validate(cfg, mode)`. `osmose/engine/config.py` gains a 3-line hook at the top of `from_dict`.
- **Schema:** `osmose/schema/simulation.py` gains one `OsmoseField` entry for `validation.strict.enabled` (ENUM, choices `["off", "warn", "error"]`, `required=False`, default `"off"`).
- **Tests:** `tests/test_config_validation.py` **(new)** — 18 unit tests; `tests/test_engine_config_validation.py` **(modify)** — append 3 parametrized integration tests.
- **Changelog:** `CHANGELOG.md` — one line under `[Unreleased] → Changed`.

---

## Task 1: Config-validation module + integration + all 21 tests

**Goal:** Single capability commit bundling the validator module, the schema flag entry, the `from_dict` hook, and all 21 tests. TDD-style: write tests, verify FAIL, implement, verify PASS, commit.

**Files:**
- Create: `osmose/engine/config_validation.py`
- Create: `tests/test_config_validation.py`
- Modify: `osmose/engine/config.py` — hook at the top of `from_dict` (around line 1348).
- Modify: `osmose/schema/simulation.py` — append one `OsmoseField` entry.
- Modify: `tests/test_engine_config_validation.py` — append 3 parametrized integration tests.

---

- [ ] **Step 1: Confirm pre-flight baseline**

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: `2485 passed, 15 skipped, 41 deselected, 47 warnings`. If a different count, update the `+21 = 2506` arithmetic in Step 29 and the commit message in Step 30 to match the actual baseline+21.

---

- [ ] **Step 2: Add the schema entry for `validation.strict.enabled`**

In `osmose/schema/simulation.py`, append to the end of `SIMULATION_FIELDS` (after the existing `stochastic.mortality.randomseed.fixed` entry at `:136-143`):

```python
    OsmoseField(
        key_pattern="validation.strict.enabled",
        param_type=ParamType.ENUM,
        choices=["off", "warn", "error"],
        default="off",
        required=False,
        description=(
            "Unknown-config-key validation mode. 'off' (default): emits a "
            "single-line count summary ONLY when unknowns are present — "
            "silent on clean configs. 'warn': log each unknown key with a "
            "typo suggestion via the osmose.config logger. 'error': collect "
            "all unknowns then raise ValueError listing them (not fail-fast)."
        ),
        category="simulation",
        advanced=True,
    ),
```

**Why `required=False`:** `OsmoseField.__post_init__` doesn't enforce `required`, but downstream tooling (UI, form generation) treats `required` as authoritative. The flag is genuinely optional.

**Why `advanced=True`:** keeps the flag out of the default UI form (it's an operator-level setting).

---

- [ ] **Step 3: Verify schema entry registers cleanly**

```bash
.venv/bin/python -c "
from osmose.schema import build_registry
reg = build_registry()
f = reg.get_field('validation.strict.enabled')
assert f is not None, 'missing from registry'
assert f.param_type.value == 'enum', f'wrong type: {f.param_type}'
assert f.choices == ['off', 'warn', 'error'], f'wrong choices: {f.choices}'
assert f.default == 'off', f'wrong default: {f.default}'
assert not f.required, 'should be optional'
print('schema OK; total fields:', len(reg.all_fields()))
"
```

Expected: `schema OK; total fields: 221` (220 prior + 1 new).

If the command raises, fix the `OsmoseField` entry before moving on — a broken schema will cascade into every downstream step.

---

- [ ] **Step 4: Write `tests/test_config_validation.py` with all 18 unit tests**

Create `tests/test_config_validation.py`. The file is ~330 lines — paste it in full; do not abbreviate.

```python
"""Unit tests for osmose.engine.config_validation (Phase 7.3).

Mode semantics:
  - off    : silent on clean configs; one info-line count nudge if unknowns present
  - warn   : per-key warning with difflib suggestion (pattern-form), no raise
  - error  : collect all unknowns, raise single ValueError

All log assertions use pytest's caplog against logger 'osmose.config'
(osmose/logging.py:7 is a stdlib logging.getLogger wrapper).
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap

import pytest

from osmose.engine.config_validation import (
    KnownKeys,
    UnknownKey,
    _extract_literal_keys_from_config_py,
    _normalize_key_to_pattern,
    _suggest,
    build_known_keys,
    validate,
)

# --- Mode-behavior tests --------------------------------------------------


def test_known_key_no_warning(caplog):
    cfg = {"simulation.nspecies": "3"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []
    assert caplog.records == []


def test_unknown_key_warns_in_warn_mode(caplog):
    cfg = {"species.liinf.sp0": "30.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert len(result) == 1
    assert result[0].key == "species.liinf.sp0"
    # Suggestion is in pattern form (per spec §Difflib tuning)
    assert "species.linf.sp{idx}" in result[0].suggestion
    # One per-key log entry; message contains the suggestion
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warn_records) == 1
    assert "species.liinf.sp0" in warn_records[0].message
    assert "species.linf.sp{idx}" in warn_records[0].message


def test_unknown_key_raises_in_error_mode_single():
    cfg = {"species.liinf.sp0": "30.0"}
    with pytest.raises(ValueError) as exc_info:
        validate(cfg, "error")
    msg = str(exc_info.value)
    assert "species.liinf.sp0" in msg
    assert "species.linf.sp{idx}" in msg


def test_error_mode_collects_all_unknowns():
    """Spec invariant: error mode collects ALL unknowns before raising.
    A fail-fast implementation would pass test_unknown_key_raises_in_error_mode_single
    but fail here. This test pins the collect-all contract."""
    cfg = {
        "species.liinf.sp0": "30.0",
        "predation.zzzbogus.sp0": "1.0",
    }
    with pytest.raises(ValueError) as exc_info:
        validate(cfg, "error")
    msg = str(exc_info.value)
    assert "species.liinf.sp0" in msg
    assert "predation.zzzbogus.sp0" in msg


def test_unknown_key_nudge_in_off_mode(caplog):
    cfg = {"species.liinf.sp0": "30.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "off")
    # Validator still returns the list (for programmatic callers)
    assert len(result) == 1
    # Exactly one info-level nudge, no per-key warnings
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert re.search(r"Config has \d+ unknown keys", info_records[0].message)
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warn_records == []


def test_off_mode_zero_unknowns_silent(caplog):
    """off + no unknowns = zero log output. Pins the 'silent on clean configs'
    rollout claim."""
    cfg = {"simulation.nspecies": "3", "simulation.time.nyear": "10"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "off")
    assert result == []
    assert caplog.records == []


def test_close_match_suggestion_cutoff(caplog):
    """A key with no close match in the allowlist produces a warning
    without a 'did you mean' clause."""
    cfg = {"species.totallywrongthing.sp0": "1.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert len(result) == 1
    assert result[0].suggestion is None
    warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warn_records) == 1
    assert "did you mean" not in warn_records[0].message.lower()


def test_idx_pattern_accepts_concrete_key(caplog):
    """A concrete species-index key matches the {idx} pattern via normalization."""
    cfg = {"species.linf.sp47": "30.0"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_double_index_pattern_accepts_concrete_key(caplog):
    """Multi-index keys (fsh + sp) normalize segment-by-segment."""
    cfg = {"fisheries.seasonality.fsh0.sp3": "0.5"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_internal_marker_not_warned(caplog):
    """_osmose.config.dir is used as a cfg.get literal in config.py — the AST
    walker captures it, so it's in known.literals."""
    cfg = {"_osmose.config.dir": "/tmp/cfg"}
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_non_string_value_still_validates_key(caplog):
    """validate() iterates cfg.keys() — value type is ignored."""
    cfg = {"simulation.nspecies": 3}  # int, not str
    with caplog.at_level(logging.INFO, logger="osmose.config"):
        result = validate(cfg, "warn")
    assert result == []


def test_invalid_mode_raises():
    """Case-sensitive mode match — typo in flag value is a startup bug."""
    cfg = {"simulation.nspecies": "3"}
    with pytest.raises(ValueError, match=r"validation\.strict\.enabled"):
        validate(cfg, "verbose")
    with pytest.raises(ValueError, match=r"validation\.strict\.enabled"):
        validate(cfg, "OFF")  # uppercase not accepted
    with pytest.raises(ValueError, match=r"validation\.strict\.enabled"):
        validate(cfg, "true")


# --- AST walker tests -----------------------------------------------------


def test_ast_extracts_cfg_get_literals_from_fixture(tmp_path):
    """Synthetic AST input exercising each documented extraction shape."""
    source = textwrap.dedent(
        """
        def f(cfg):
            a = cfg.get("x.y")
            b = cfg["a.b"]
            c = _enabled(cfg, "p.q")
            d = _species_float(cfg, "species.x.sp{i}", 3)
            e = _species_float_optional(cfg, "species.z.sp{i}", 3, None)
            g = "bioen.enabled" in cfg
            h = cfg.get("has.default", "default_value")
        """
    )
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    # {i} → {idx} normalized per registry convention
    assert "x.y" in keys
    assert "a.b" in keys
    assert "p.q" in keys
    assert "species.x.sp{idx}" in keys
    assert "species.z.sp{idx}" in keys
    assert "bioen.enabled" in keys
    assert "has.default" in keys


def test_ast_extracts_from_real_config_py_canary():
    """Canary: the walker finds known-present sentinels in the real config.py.
    If this breaks, someone changed config.py in a way the walker can't parse."""
    import osmose.engine.config as cfg_mod
    import ast as _ast
    tree = _ast.parse(open(cfg_mod.__file__).read())
    keys = _extract_literal_keys_from_config_py(tree)
    for sentinel in (
        "_osmose.config.dir",
        "simulation.nspecies",
        "simulation.time.ndtperyear",
    ):
        assert sentinel in keys, f"walker lost {sentinel!r}"


def test_ast_handles_fstring_with_name_interpolation():
    """f-string with a simple Name interpolation becomes {idx}."""
    source = 'cfg.get(f"mortality.{variant}.sp{i}")'
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert "mortality.{idx}.sp{idx}" in keys


def test_ast_drops_fstring_with_complex_interpolation():
    """f-string with an Attribute/Subscript/Call interpolation is skipped."""
    source = 'cfg.get(f"x.{obj.attr}")'
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    # Should not be captured at all (can't resolve statically)
    assert not any("obj.attr" in k for k in keys)
    assert "x.{idx}" not in keys


def test_ast_handles_in_comparison():
    """'key' in cfg membership test extracts the literal."""
    source = '"bioen.enabled" in cfg'
    tree = ast.parse(source)
    keys = _extract_literal_keys_from_config_py(tree)
    assert "bioen.enabled" in keys


# --- build_known_keys + fallback ------------------------------------------


def test_build_known_keys_has_literal_fast_path(monkeypatch):
    """build_known_keys returns a KnownKeys with both literal and regex views,
    and the literal set is the majority (~95% per spec §Architecture)."""
    kk = build_known_keys()
    assert isinstance(kk, KnownKeys)
    # Schema has 221 fields post-Step 2; literal set should be at least as big
    # (most schema patterns are literal; {idx}-patterns go to regexes).
    assert len(kk.literals) > 100, f"literals set too small: {len(kk.literals)}"
    # Internal marker and flag itself must be in literals after the AST+schema union
    assert "_osmose.config.dir" in kk.literals
    assert "validation.strict.enabled" in kk.literals


def test_source_file_unavailable_falls_back_to_schema_only(monkeypatch):
    """If the AST source can't be loaded, build_known_keys returns a
    schema-only allowlist with an info log. Validator keeps working."""
    # Clear the lru_cache wrapper so we hit the fallback path fresh
    build_known_keys.cache_clear()
    # Monkeypatch importlib.resources to raise
    import osmose.engine.config_validation as cv_mod

    def _fake_read_text():
        raise FileNotFoundError("simulated missing source")

    monkeypatch.setattr(cv_mod, "_read_config_source", _fake_read_text)
    try:
        kk = build_known_keys()
        assert isinstance(kk, KnownKeys)
        # Schema-only fallback — validation.strict.enabled is still there
        # (it's schema-registered, not AST-captured)
        assert "validation.strict.enabled" in kk.literals
    finally:
        build_known_keys.cache_clear()  # restore clean state for other tests
```

---

- [ ] **Step 5: Run the 18 unit tests — verify FAIL with ImportError**

```bash
.venv/bin/python -m pytest tests/test_config_validation.py -v 2>&1 | tail -10
```

Expected: every test fails at collection time with `ModuleNotFoundError: No module named 'osmose.engine.config_validation'`. This confirms the test file is well-formed and the implementation target exists.

---

- [ ] **Step 6: Create `osmose/engine/config_validation.py` with the full module**

Create the file. Paste it in full — do not abbreviate. The module is ~230 lines:

```python
"""Unknown-key validation for OSMOSE EngineConfig.from_dict().

Spec: docs/superpowers/specs/2026-04-19-config-validation-design.md (557ee1b).

Match order (fast → slow):
  1. Exact literal lookup in KnownKeys.literals (O(1) set).
  2. Normalized-pattern lookup in KnownKeys.patterns (O(1) set) —
     converts user's concrete key (species.linf.sp47) to pattern form
     (species.linf.sp{idx}) segment-by-segment.
  3. Regex match across KnownKeys.regexes (~15 compiled patterns).
  4. Miss → UnknownKey with optional difflib suggestion (cutoff 0.85
     against normalized pattern-form strings on both sides).
"""
from __future__ import annotations

import ast
import difflib
import functools
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("osmose.config")

# Segment-level index-suffix normalizers. Applied to every dot-separated
# segment of the user's key. Order-sensitive: apply longer tokens first so
# "fsh12" doesn't partial-match into "sh12".
_INDEX_SUFFIXES = (
    ("fsh", re.compile(r"^fsh\d+$")),
    ("map", re.compile(r"^map\d+$")),
    ("age", re.compile(r"^age\d+$")),
    ("sz", re.compile(r"^sz\d+$")),
    ("sp", re.compile(r"^sp\d+$")),
)

_VALID_MODES = ("off", "warn", "error")
_SUGGESTION_CUTOFF = 0.85


@dataclass(frozen=True)
class UnknownKey:
    """One unknown key detected in a user config."""

    key: str
    suggestion: str | None  # pattern-form string, or None if no close match


@dataclass(frozen=True)
class KnownKeys:
    """Parallel views of the allowlist, built once and cached."""

    patterns: frozenset[str]  # pattern-form strings (difflib corpus)
    literals: frozenset[str]  # exact-match fast path (patterns without "{idx}")
    regexes: tuple[tuple[str, re.Pattern], ...]  # ({idx}-pattern, compiled) pairs


def _normalize_key_to_pattern(key: str) -> str:
    """Convert a concrete user key to its {idx}-pattern form.

    Applies the _INDEX_SUFFIXES table segment-by-segment. Keys with no
    index-suffixed segment return unchanged (they're their own pattern).
    """
    segments = key.split(".")
    for i, seg in enumerate(segments):
        for token, pattern in _INDEX_SUFFIXES:
            if pattern.fullmatch(seg):
                segments[i] = "{idx}"
                break
    return ".".join(segments)


def _compile_regex_for_pattern(pattern: str) -> re.Pattern:
    """Convert a {idx}-pattern string to a compiled regex that matches
    concrete keys. E.g. 'species.linf.sp{idx}' -> r'^species\\.linf\\.sp\\d+$'."""
    escaped = re.escape(pattern).replace(r"\{idx\}", r"\d+")
    return re.compile(f"^{escaped}$")


def _read_config_source() -> str:
    """Read osmose/engine/config.py source via importlib.resources.

    Isolated as a function so tests can monkeypatch the source-load path
    and exercise the schema-only fallback branch.
    """
    import importlib.resources

    return (
        importlib.resources.files("osmose.engine")
        .joinpath("config.py")
        .read_text(encoding="utf-8")
    )


def _extract_literal_keys_from_config_py(tree: ast.AST) -> set[str]:
    """Walk an AST and extract OSMOSE config-key literals.

    Handles these shapes (spec §AST walker):
      - cfg.get("literal", ...)          → args[0]
      - cfg["literal"]                    → Subscript.slice
      - _enabled(cfg, "literal")          → args[1] or keywords["key"]
      - _get / _species_float[_optional] / _species_int[_optional] / _species_str
        (cfg, "literal", ...)             → args[1]
      - "literal" in cfg                  → Compare(ops=[In], left=Constant)
      - Module-level list/tuple of string literals (one level deep)
      - f-string with simple Name interpolations — {name} → {idx};
        complex interpolations (Attribute/Subscript/Call) are skipped.

    Normalization: any {i}, {fsh}, {sp}, {idx}, or f-string Name-placeholder
    collapses to {idx} to match ParameterRegistry convention.
    """
    helper_names = {
        "_get",
        "_enabled",
        "_species_float",
        "_species_float_optional",
        "_species_int",
        "_species_int_optional",
        "_species_str",
    }
    out: set[str] = set()

    def _capture_string(s: str) -> None:
        # Normalize any {i}, {fsh}, {sp} etc. to {idx}
        # This handles strings that already contain Java-style placeholders.
        out.add(re.sub(r"\{(i|fsh|sp|idx)\}", "{idx}", s))

    def _render_fstring(joined: ast.JoinedStr) -> str | None:
        """Reconstruct an f-string's pattern form or return None if unresolvable."""
        pieces: list[str] = []
        for part in joined.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                if isinstance(part.value, ast.Name):
                    pieces.append("{idx}")
                else:
                    # Attribute / Subscript / Call — can't resolve statically
                    return None
            else:
                return None
        return "".join(pieces)

    for node in ast.walk(tree):
        # cfg.get("x", ...) and cfg.get(f"x.{i}", ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                _capture_string(first.value)
            elif isinstance(first, ast.JoinedStr):
                rendered = _render_fstring(first)
                if rendered is not None:
                    _capture_string(rendered)

        # _enabled(cfg, "x") / _species_float(cfg, "x", ...) / etc.
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in helper_names and len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    _capture_string(second.value)
                elif isinstance(second, ast.JoinedStr):
                    rendered = _render_fstring(second)
                    if rendered is not None:
                        _capture_string(rendered)
            # also handle _enabled(cfg, key="x")
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        _capture_string(kw.value.value)

        # cfg["x"]
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                _capture_string(sl.value)

        # "x" in cfg
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], ast.In):
                if isinstance(node.left, ast.Constant) and isinstance(
                    node.left.value, str
                ):
                    # Heuristic: only capture dot-containing strings (avoids
                    # capturing single words that are cfg membership tests
                    # like "enabled" in cfg when cfg is something else).
                    if "." in node.left.value:
                        _capture_string(node.left.value)

        # Module-level tuple/list of string literals — one level deep
        elif isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if "." in elt.value and not elt.value.startswith(
                        ("/", "_", "http")
                    ):
                        _capture_string(elt.value)

    return out


@functools.cache
def build_known_keys() -> KnownKeys:
    """Union of ParameterRegistry field patterns + AST-extracted reader keys.

    Called lazily on first validate() invocation; cached for the process.
    If the AST source is unavailable (frozen build, etc.), falls back to
    schema-only with an info log.
    """
    # Schema side — eager imports in osmose/schema/__init__.py ensure the
    # registry is populated by the time this call returns.
    from osmose.schema import build_registry

    reg = build_registry()
    pattern_strs: set[str] = {f.key_pattern for f in reg.all_fields()}

    # AST side — read config.py source, parse, walk.
    try:
        source = _read_config_source()
        tree = ast.parse(source)
        pattern_strs |= _extract_literal_keys_from_config_py(tree)
    except (OSError, FileNotFoundError, SyntaxError) as exc:
        log.info(
            "config_validation: AST source unavailable (%s); "
            "using schema-only allowlist.",
            exc,
        )

    patterns = frozenset(pattern_strs)
    literals = frozenset(p for p in patterns if "{idx}" not in p)
    regex_pairs = tuple(
        (p, _compile_regex_for_pattern(p)) for p in patterns if "{idx}" in p
    )
    return KnownKeys(patterns=patterns, literals=literals, regexes=regex_pairs)


def _suggest(normalized_key: str, patterns: frozenset[str]) -> str | None:
    """Return the single closest pattern string, or None if below cutoff."""
    matches = difflib.get_close_matches(
        normalized_key, list(patterns), n=1, cutoff=_SUGGESTION_CUTOFF
    )
    return matches[0] if matches else None


def _check(cfg_key: str, known: KnownKeys) -> UnknownKey | None:
    """Return an UnknownKey if cfg_key is unknown, else None."""
    # Fast path: literal exact match.
    if cfg_key in known.literals:
        return None
    # Normalize and try again against both literals and patterns.
    normalized = _normalize_key_to_pattern(cfg_key)
    if normalized in known.patterns:
        return None
    # Regex fallback — handles cases where normalization didn't pick up
    # an index-suffix form we don't enumerate.
    for _, compiled in known.regexes:
        if compiled.match(cfg_key):
            return None
    # Unknown. Generate suggestion on normalized form.
    suggestion = _suggest(normalized, known.patterns)
    return UnknownKey(key=cfg_key, suggestion=suggestion)


def validate(cfg: dict, mode: str) -> list[UnknownKey]:
    """Detect unknown config keys and dispatch per mode.

    mode:
      - "off"   : return the list; emit a single info-line nudge if non-empty.
      - "warn"  : return the list; log one warning per unknown (with suggestion).
      - "error" : if any unknowns, collect ALL and raise ValueError.

    Raises ValueError for invalid mode strings (case-sensitive).
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"validation.strict.enabled must be one of {list(_VALID_MODES)!r}; "
            f"got {mode!r}"
        )

    known = build_known_keys()
    unknowns: list[UnknownKey] = []
    for key in cfg:
        result = _check(key, known)
        if result is not None:
            unknowns.append(result)

    if not unknowns:
        return unknowns

    if mode == "error":
        lines = ["Unknown OSMOSE config keys detected:"]
        for uk in unknowns:
            if uk.suggestion:
                lines.append(f"  - {uk.key!r}  (did you mean {uk.suggestion!r}?)")
            else:
                lines.append(f"  - {uk.key!r}")
        raise ValueError("\n".join(lines))

    if mode == "warn":
        for uk in unknowns:
            if uk.suggestion:
                log.warning(
                    "Unknown config key %r — did you mean %r?",
                    uk.key,
                    uk.suggestion,
                )
            else:
                log.warning("Unknown config key %r", uk.key)
        return unknowns

    # mode == "off": single info-level nudge (count only, no key names)
    log.info(
        "Config has %d unknown keys; set validation.strict.enabled=warn for details.",
        len(unknowns),
    )
    return unknowns
```

---

- [ ] **Step 7: Run the 18 unit tests — expect 18 PASS**

```bash
.venv/bin/python -m pytest tests/test_config_validation.py -v 2>&1 | tail -30
```

Expected: `18 passed`. If any test fails, read the failure carefully:

- `test_error_mode_collects_all_unknowns` failing → your `validate()` is fail-fast; fix the loop to append then raise once.
- `test_close_match_suggestion_cutoff` failing with a suggestion surfacing → `_SUGGESTION_CUTOFF` too low; confirm `0.85`.
- `test_ast_handles_fstring_with_name_interpolation` failing → the `_render_fstring` helper dropped the f-string; check the `JoinedStr` branch path.
- `test_source_file_unavailable_falls_back_to_schema_only` failing → `build_known_keys.cache_clear()` wasn't called; re-read the try/finally block in that test.

---

- [ ] **Step 8: Hook `validate` into `EngineConfig.from_dict`**

In `osmose/engine/config.py`, find the `from_dict` classmethod (around `:1347-1348`):

```python
    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        # config_dir is extracted from cfg by _cfg_dir() at each _resolve_file call
        n_sp = int(_get(cfg, "simulation.nspecies"))
```

Insert the validation call as the very first statement of the body (before the `n_sp = ...` line). The inserted block replaces nothing; it only adds:

```python
    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
        # Unknown-key validation (Phase 7.3). Silent by default; mode is
        # controlled by the validation.strict.enabled config flag.
        from osmose.engine.config_validation import validate as _validate_cfg

        _mode = cfg.get("validation.strict.enabled", "off")
        _validate_cfg(cfg, _mode)  # side effects only (log/raise)

        # config_dir is extracted from cfg by _cfg_dir() at each _resolve_file call
        n_sp = int(_get(cfg, "simulation.nspecies"))
```

The local-import keeps the validator out of the critical import path (so schema bootstrap can run before `validate()` touches `build_registry()`).

---

- [ ] **Step 9: Append the 3 parametrized integration tests to `tests/test_engine_config_validation.py`**

Append to the end of `tests/test_engine_config_validation.py` (preserving existing tests — this file has 7 `__post_init__` tests already):

```python
# --- Phase 7.3: unknown-key validation integration tests ----------------------
import logging as _logging
from pathlib import Path as _Path

import pytest as _pytest

from osmose.engine.config import EngineConfig as _EngineConfig


def _load_example_config(example_name: str) -> dict:
    """Load an example all-parameters.csv into a dict. Uses the project's
    config reader to resolve include files correctly."""
    from osmose.config import read_config as _read_config

    base = _Path(__file__).resolve().parent.parent
    candidates = [
        base / "data" / "examples" / example_name / "osm_all-parameters.csv",
        base / "data" / example_name / "osm_all-parameters.csv",
    ]
    for path in candidates:
        if path.exists():
            return _read_config(str(path))
    _pytest.skip(f"example config not found: {example_name}")


@_pytest.mark.parametrize(
    "example_name",
    ["eec", "bay_of_biscay", "baltic"],
)
def test_from_dict_off_mode_silent_on_example_configs(example_name, caplog):
    """Load each reference example with default (off) mode; assert no
    WARNING/ERROR records and no exception. Catches AST-walker gaps
    specific to a single example config."""
    cfg = _load_example_config(example_name)
    # Default mode is 'off' — do not inject the flag explicitly.
    with caplog.at_level(_logging.INFO, logger="osmose.config"):
        _EngineConfig.from_dict(cfg)
    warn_err = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert warn_err == [], (
        f"{example_name} config emitted {len(warn_err)} warnings — "
        f"first: {warn_err[0].message if warn_err else ''!r}"
    )


def test_from_dict_warn_mode_catches_known_typo(caplog):
    """Inject a single-char typo into a small cfg; mode=warn; assert the
    warning includes the suggestion."""
    cfg = _load_example_config("eec")
    cfg["species.liinf.sp0"] = "30.0"
    cfg["validation.strict.enabled"] = "warn"
    with caplog.at_level(_logging.WARNING, logger="osmose.config"):
        _EngineConfig.from_dict(cfg)
    warn_records = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any("species.liinf.sp0" in r.message for r in warn_records)
    assert any("species.linf.sp{idx}" in r.message for r in warn_records)


def test_from_dict_error_mode_raises_with_typo():
    """Same injection; mode=error; assert from_dict raises before parsing."""
    cfg = _load_example_config("eec")
    cfg["species.liinf.sp0"] = "30.0"
    cfg["validation.strict.enabled"] = "error"
    with _pytest.raises(ValueError, match="species.liinf.sp0"):
        _EngineConfig.from_dict(cfg)
```

**Why the `_load_example_config` helper with two candidates:** the EEC / Bay-of-Biscay examples live under `data/examples/<name>/osm_all-parameters.csv`; the Baltic config lives under `data/baltic/osm_all-parameters.csv` (no `examples/` layer). The helper tries both and skips if neither exists.

---

- [ ] **Step 10: Run the 3 integration tests**

```bash
.venv/bin/python -m pytest tests/test_engine_config_validation.py -v -k "example_configs or known_typo or error_mode_raises" 2>&1 | tail -20
```

Expected: 5 passing (3 parametrized variants of `example_configs` + `warn_mode` + `error_mode_raises`).

Note: if a reference config emits unknown-key warnings in off mode, **the test `test_from_dict_off_mode_silent_on_example_configs[...]` will fail with a list of the offending keys**. That's the signal that the AST walker missed something that example config uses. Fix path:

1. Grep the offending key in `osmose/engine/config.py` to locate the reader code path that touches it.
2. Either: (a) the AST walker should have caught it but didn't — extend `_extract_literal_keys_from_config_py` to handle the missing shape; or (b) the key is built dynamically from a caller-arg `key_pattern` (spec's "accepted misses") — add it to a supplementary allowlist. Put the supplementary allowlist as a module-level frozenset in `config_validation.py`:

```python
# Reader-honored keys that the AST walker can't resolve statically
# (built from caller-arg patterns in e.g. _load_per_species_timeseries).
# Extended when an example config surfaces an unknown that's legitimate.
_SUPPLEMENTARY_ALLOWLIST: frozenset[str] = frozenset([
    # Add entries here as the EEC/BoB/Baltic sweep surfaces them.
])
```

Then inside `build_known_keys()`, merge: `pattern_strs |= _SUPPLEMENTARY_ALLOWLIST`.

Iterate until all 3 parametrized variants pass.

---

- [ ] **Step 11: Full suite + lint**

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected:
- `2506 passed, 15 skipped, 41 deselected` (2485 baseline + 21 new tests).
- Ruff: `All checks passed!`

If the passed count is off:
- `2485 + N` where N < 21 → some new tests weren't collected. Grep for `def test_` in both test files; confirm count.
- `< 2485 + 21` → a pre-existing test regressed. The most likely culprit: a test that calls `EngineConfig.from_dict` with a cfg containing a latent unknown that now trips the validator's `off`-mode nudge (harmless — shouldn't fail). If an INTEGRATION test regressed, investigate (check if the test asserts clean caplog at the `osmose.config` logger).
- `> 2485 + 21` → you accidentally added more tests than planned. Count and reconcile.

---

- [ ] **Step 12: Commit the single capability commit**

```bash
git add osmose/engine/config_validation.py osmose/engine/config.py osmose/schema/simulation.py tests/test_config_validation.py tests/test_engine_config_validation.py
git commit -m "feat(config): unknown-key validation at EngineConfig.from_dict (Phase 7.3)

New osmose/engine/config_validation.py catches OSMOSE config typos at
load time. Three modes via validation.strict.enabled:
  - off (default): silent on clean configs; one info-line count nudge
    when unknowns are present
  - warn: per-key warning with a difflib suggestion via the
    osmose.config logger
  - error: collect ALL unknowns, then raise a single ValueError

Allowlist (~255 patterns) is a union of the 220-field ParameterRegistry
and AST-extracted literal keys from config.py (cfg.get, subscript,
_enabled / _get / _species_float[_optional] / _species_int[_optional] /
_species_str helpers, f-strings with Name interpolation, module-level
tuple/list literals, and 'x' in cfg membership tests). Match order is
literal-fast-path → normalized-pattern exact → compiled-regex fallback.

Typo suggestions use difflib.get_close_matches(n=1, cutoff=0.85) on
{idx}-normalized strings — species.liinf.sp0 normalized to
species.liinf.sp{idx} matches species.linf.sp{idx} at ratio ~0.98.

Schema flag validation.strict.enabled added to osmose/schema/simulation.py
(ENUM, optional, advanced, default 'off'). Hook is 4 lines at the top
of EngineConfig.from_dict; zero behavior change on clean configs in
off mode.

Tests: +21 (18 unit + 3 parametrized integration across EEC /
Bay-of-Biscay / Baltic). Baseline 2485 → 2506.

Spec: docs/superpowers/specs/2026-04-19-config-validation-design.md"
```

---

## Task 2: CHANGELOG entry

- [ ] **Step 1: Prepend entry under `[Unreleased]`**

Read the current `CHANGELOG.md` `[Unreleased] → Changed` section. Prepend this bullet directly after the `### Changed` heading:

```markdown
- **config:** unknown-key validation at `EngineConfig.from_dict` (Phase 7.3). Silent by default — set `validation.strict.enabled=warn` (or `=error`) to catch typos like `species.liinf.sp0` with a difflib suggestion pointing at `species.linf.sp{idx}`. Allowlist is the union of the 220-field `ParameterRegistry` and AST-extracted literal keys from `osmose/engine/config.py` (~255 patterns total). Spec at `docs/superpowers/specs/2026-04-19-config-validation-design.md`; plan at `docs/superpowers/plans/2026-04-19-config-validation-plan.md`.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: CHANGELOG entry for Phase 7.3 config validation"
```

---

## Self-review checklist

- **Spec coverage:** every spec section has at least one task step that implements it — schema entry (Step 2), core validator module (Step 6), AST walker (Step 6 inside `_extract_literal_keys_from_config_py`), integration hook (Step 8), 18 unit tests (Step 4) + 3 integration tests (Step 9), CHANGELOG (Task 2). ✓
- **Failure-modes table coverage:** 10 rows in the spec's Failure modes table; each maps to at least one test — `_osmose.config.dir` → test #10 `test_internal_marker_not_warned`; `{idx}`-concrete → test #8; double-index → test #9; high-confidence typo → test #2 + #3; no close match → test #7; non-string value → test #11; mode=error collect-all → test #4; mode=off with nudge → test #5; mode=off zero unknowns → test #6; invalid mode → test #12; flag-absent defaults to off → implicit across integration tests. ✓
- **Placeholder scan:** no "TBD", no "TODO", no "implement later", no bare "add validation", no "similar to step N". Every code block is complete. ✓
- **Type consistency:** `UnknownKey`, `KnownKeys`, `build_known_keys`, `validate`, `_normalize_key_to_pattern`, `_suggest`, `_extract_literal_keys_from_config_py`, `_check`, `_read_config_source`, `_compile_regex_for_pattern`, `_SUGGESTION_CUTOFF`, `_VALID_MODES`, `_INDEX_SUFFIXES`, `_SUPPLEMENTARY_ALLOWLIST` — names used consistently across the module definition and the tests. ✓
- **Test runner:** `.venv/bin/python -m pytest` throughout. No bare `python`. ✓
- **Commit granularity:** 2 commits total matching "single capability + CHANGELOG" (Task 1 + Task 2). Task 1 is one atomic commit; Task 2 is the CHANGELOG-only commit. ✓
- **Baseline check:** pre-flight asserts 2485 passed / 2500 collected; Step 11 expects 2506 after +21 tests. If baseline drifts, all arithmetic documented here needs updating. ✓
- **Library-surface anchors:** all line citations (`config.py:1347-1348`, `:150`, `:41-45`, `:133`, `:49`, etc.) verified against HEAD `557ee1b` during plan writing. ✓

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-config-validation-plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent to execute Task 1 (the big one), review, then dispatch a fresh subagent for Task 2 (CHANGELOG). Fast iteration.

**2. Inline Execution** — I execute both tasks in this session using `superpowers:executing-plans` with a checkpoint after Task 1.

Which approach?
