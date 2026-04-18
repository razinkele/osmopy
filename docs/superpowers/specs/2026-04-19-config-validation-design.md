# Config Validation — Design

> **Roadmap item:** `docs/parity-roadmap.md` Phase 7.3. Spec for the plan at (forthcoming) `docs/superpowers/plans/2026-04-19-config-validation-plan.md`.

**Goal.** Catch OSMOSE config typos at load time instead of letting them produce silent downstream garbage. Today a misspelled key (`species.liinf.sp0` with an extra `i`) is silently ignored by the reader; the simulation runs with the default value and the user only finds out when the numbers look wrong. This spec adds a validator that flags unknown keys with optional typo suggestions and a mode flag for rollout safety.

**Allowlist size (verified HEAD `e8294ee`):** `ParameterRegistry.build_registry()` exposes 220 fields. A regex sweep of `osmose/engine/config.py` finds 134 reader-referenced string-literal keys. After `{idx}`-normalization and cross-matching, 99 of the 134 reader keys are already covered by the schema registry; the remaining 35 reader-only keys (`evolution.*`, `fisheries.*`, `economic.*`, the internal `_osmose.config.dir`, etc.) are net-new for the AST walker to contribute. Union size ≈ 220 + 35 = 255 patterns, which split into ~240 literals (fast path) and ~15 `{idx}`-patterns (regex path).

## Scope choice

- **In scope:** unknown-key detection at `EngineConfig.from_dict` time. Three opt-in modes (`off` / `warn` / `error`). AST-parsed reader allowlist unioned with the 220-field `ParameterRegistry`. Typo suggestions via `difflib`.
- **Out of scope (v0.9.2):** missing-required-key checks (the roadmap entry mentions both; we cover unknowns only to keep the v0.9.2 diff narrow). Auto-fix. IDE/LSP integration. Per-module validation profiles. Value-level validation beyond what `ParameterRegistry.validate()` already does.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  EngineConfig.from_dict(cfg)    # osmose/engine/config.py:1348  │
│    ↓                                                            │
│  config_validation.validate(cfg, mode)    # NEW MODULE          │
│    ├─ build_known_keys()  (cached, once per process)            │
│    │    ├─ ParameterRegistry  (220 schema fields)               │
│    │    └─ AST parse of config.py                               │
│    │         └─ cfg.get("x"), cfg["x"], _enabled(cfg, "x")      │
│    ├─ diff: cfg.keys() \ known                                  │
│    ├─ for each unknown: difflib.get_close_matches(n=1, ≥0.8)    │
│    └─ dispatch by mode:                                         │
│         off   → count-only summary (1 line via osmose.logging)  │
│         warn  → per-key log + suggestions                       │
│         error → collect all, raise ValueError with full list    │
│    ↓                                                            │
│  (existing from_dict parse body)                                │
└─────────────────────────────────────────────────────────────────┘
```

No library-side changes beyond:
1. New module `osmose/engine/config_validation.py`.
2. Hook-call at the top of `EngineConfig.from_dict` (one line).
3. New schema entry for `validation.strict.enabled` in `osmose/schema/simulation.py`.

## Components

### `osmose/engine/config_validation.py` (new)

```python
@dataclass(frozen=True)
class UnknownKey:
    key: str
    suggestion: str | None   # populated only when difflib confidence ≥ cutoff

@dataclass(frozen=True)
class KnownKeys:
    """Two parallel views of the allowlist so match-path and suggestion-path
    each use the right data structure. Built once, cached via functools.cache.
    """
    patterns: frozenset[str]              # original pattern strings (for difflib)
    literals: frozenset[str]              # subset containing no "{idx}" — fast path
    regexes: tuple[tuple[str, re.Pattern], ...]   # (pattern_str, compiled_regex) pairs for {idx}-patterns only

def build_known_keys() -> KnownKeys:
    """Union of (ParameterRegistry field patterns) + (AST-extracted reader keys).

    Each pattern string is normalized so any single-name interpolation
    ({i}, {fsh}, {sp}, {idx}) collapses to {idx}, matching the
    ParameterRegistry convention. Regexes compile only for patterns
    that contain {idx} after normalization — literal patterns go to
    ``literals`` for exact-match fast path.

    Cached lazily on first call via functools.cache.
    """

def validate(cfg: dict[str, str], mode: str) -> list[UnknownKey]:
    """Return the list of unknown keys (empty = all known).

    Match order (fast → slow):
      1. Key ∈ known.literals → known (O(1) set-lookup)
      2. Any compiled regex in known.regexes matches → known
      3. Otherwise → unknown. Suggestion = ``_suggest(key, known.patterns)``
         (normalize key to {idx} form first; see "Difflib tuning" below).

    ``mode`` controls side effects:
      - "off"   → return list, emit single-line count summary if nonempty
      - "warn"  → return list, log each unknown + suggestion via
                  logging.getLogger("osmose.config").warning(...)
      - "error" → if list nonempty, collect ALL unknowns then raise a
                  single ValueError listing them (not fail-fast).

    Raises ValueError on invalid mode strings ("on", "true", "enabled", …).
    Mode comparison is case-sensitive — typos in the flag value itself
    are a startup bug, not a soft warning.
    """

def _normalize_key_to_pattern(key: str) -> str:
    """Convert a concrete user key to its {idx}-pattern form.

    Replaces any trailing ``sp\\d+`` / ``fsh\\d+`` with the corresponding
    ``{idx}`` token, plus a handful of other single-index suffixes
    (``.map\\d+``, ``.age\\d+``, ``.sz\\d+``). Used both for pattern
    matching in the ``regexes`` path and for prepping the user's key
    before difflib comparison.

    Concrete normalization table is a small static list committed in
    this module so the implementer can grep and extend.
    """

def _suggest(user_key_normalized: str, patterns: frozenset[str]) -> str | None:
    """Return the single closest pattern-string suggestion, or None.

    Uses ``difflib.get_close_matches(user_key_normalized, patterns,
    n=1, cutoff=0.85)``. Operates on normalized (pattern-form) strings
    on both sides — so ``species.liinf.sp{idx}`` vs
    ``species.linf.sp{idx}`` ratio ≈ 0.98, well above cutoff.
    """
```

### AST walker

One module-local function `_extract_literal_keys_from_config_py() -> set[str]`.

**Source loading.** Prefer `importlib.resources.files("osmose.engine").joinpath("config.py").read_text()` over `pathlib.Path(__file__).with_name("config.py").read_text()` — resource-loader access works uniformly across editable installs, wheel installs, and zipapp/frozen builds; `__file__` can be `None` or point inside a zip under a namespace loader. If the resource is unavailable (rare), catch `OSError`/`FileNotFoundError` and fall back to **schema-only allowlist** with an `info`-level log noting the degraded mode. The validator must never crash startup because the AST parse fails.

**Node visitation.** Walk with `ast.walk` and match these shapes:

| AST shape | Example in config.py | Extraction rule |
|---|---|---|
| `cfg.get(Constant, ...)` | `cfg.get("species.linf.sp0", ...)` | Capture `args[0].value` |
| `cfg[Constant]` (Subscript) | `cfg["simulation.nyear"]` | Capture `slice.value` |
| `_enabled(cfg, Constant)` / `_enabled(cfg, key=Constant)` | `_enabled(cfg, "output.spatial.enabled")` | Capture `args[1].value` or `keywords[key].value` |
| `_get(cfg, Constant)` / `_species_float(cfg, Constant, ...)` / `_species_float_optional(cfg, Constant, ...)` / `_species_int(cfg, Constant, ...)` / `_species_int_optional(cfg, Constant, ...)` / `_species_str(cfg, Constant, ...)` | `_species_float(cfg, "species.k.sp{i}", n_sp)` | Capture `args[1].value` |
| `Constant in cfg` (Compare with `In` op) | `"bioen.enabled" in cfg` | Capture `left.value` |
| Module-level assignments of `List`/`Tuple` of `Constant` (str) elements | `_FISHING_SCENARIOS = [("x", True), ...]` | Capture every string element reachable from module-scope `ast.Assign` nodes whose value is a nested literal collection. Non-recursive — one level deep — to avoid false captures from other string lists. |
| `JoinedStr` (f-string) | `f"mortality.{variant}.sp{i}"` | **Reconstruct the pattern**: every `FormattedValue` whose `value` is a simple `Name` (loop variable) becomes the token `{idx}`; escape literal segments. `{variant}` → `{idx}`, `{i}` → `{idx}`. Drop (skip) the whole f-string if any `FormattedValue.value` is a complex expression (`Call`, `Attribute`, `Subscript`) — we cannot statically resolve it. |
| `Constant.format(**kwargs)` call passed to `cfg.get` | `cfg.get("species.x.sp{i}".format(i=idx))` | Capture `Constant.value` as-is. `{i}` is normalized below. |

**Accepted misses (documented limitations).** Keys built from a `key_pattern` function argument (e.g. `cfg.get(key_pattern.format(i=i), "")` inside `_load_per_species_timeseries`) cannot be resolved statically — the caller is 2+ stack frames away. These keys are covered **on the schema side** if the caller passes a registered pattern; otherwise they land in the "unknown to validator but still honored by reader" bucket and surface as false-positive warnings. Acceptable for v1; the plan's rollout allowlist can capture the handful of affected keys during the EEC/Baltic/Bay-of-Biscay exercise.

**Normalization.** Keys containing `{i}`, `{fsh}`, `{sp}`, `{idx}` or `{variant}` (or any f-string interpolation of a simple `Name`) are all collapsed to `{idx}`, matching the `ParameterRegistry` convention.

**Caching.** Result cached **lazily on first call** via `@functools.cache`, not at module import — avoids import-order fragility with `ParameterRegistry` submodule registration. Inside `build_known_keys()` the import is local (`from osmose.schema import build_registry`) to force `osmose/schema/__init__.py`'s eager submodule imports before the registry snapshot is read. The first `EngineConfig.from_dict()` call pays a one-time AST-parse cost (~10-15ms measured on 1894-line config.py); subsequent calls are free.

**Verified helper names** (per a ground-truth check of `osmose/engine/config.py` at HEAD `e8294ee`): `_get`, `_enabled`, `_species_float`, `_species_float_optional`, `_species_int`, `_species_int_optional`, `_species_str`. There is no `_species_string`, `_species_array`, or `_species_bool` — an earlier draft of this spec listed those; they don't exist.

### Integration point

In `osmose/engine/config.py:1348`:

```python
@classmethod
def from_dict(cls, cfg: dict[str, str]) -> EngineConfig:
    # NEW: validate before expensive parse
    from osmose.engine.config_validation import validate as _validate_cfg
    mode = cfg.get("validation.strict.enabled", "off")
    _validate_cfg(cfg, mode)  # side effects (log/raise); return value ignored here

    # ... existing parse body ...
```

`mode` defaults to `"off"` when the flag is absent — zero behavior change for existing configs.

### Schema entry

In `osmose/schema/simulation.py`, add:

```python
OsmoseField(
    key_pattern="validation.strict.enabled",
    param_type=ParamType.ENUM,
    choices=["off", "warn", "error"],   # list per OsmoseField type annotation
    default="off",
    category="simulation",
    description=(
        "Unknown-config-key validation mode. 'off' (default): emits a "
        "single-line count summary ONLY when unknowns are present — "
        "silent on clean configs. 'warn': log each unknown key with a "
        "typo suggestion via the osmose.config logger. 'error': collect "
        "all unknowns then raise ValueError listing them (not fail-fast)."
    ),
),
```

This makes the validator recognize its own flag (avoids a self-referential "unknown key" warning for `validation.strict.enabled`).

## Failure modes and semantics

Match order inside `validate()` is fast → slow:
1. **Exact literal match** against `known.literals` (O(1) set lookup). Covers ~95% of keys on a typical cfg (`_osmose.config.dir`, `simulation.nspecies`, any non-`{idx}` key).
2. **Normalized pattern match**: compute `_normalize_key_to_pattern(key)`, check if normalized key ∈ `known.patterns`. Same O(1), handles `species.linf.sp47` via its `species.linf.sp{idx}` form.
3. **Regex fallback** across `known.regexes` (≤ ~50 `{idx}`-compiled patterns). Catches cases where the normalizer missed an index-suffix form we didn't enumerate (e.g., a rare Java-legacy index token).
4. If all three fail → unknown. Compute suggestion on the normalized key.

| Situation | Behavior |
|---|---|
| `cfg` contains `_osmose.config.dir` (internal marker) | Step 1 succeeds — it's a literal captured by the AST walker. |
| `cfg` has a `{idx}`-concrete key like `species.linf.sp47` | Step 2 succeeds — normalized to `species.linf.sp{idx}` which is in `known.patterns`. |
| `cfg` has a double-index key like `fishery.catchability.fsh0.sp3` | Step 2 succeeds — normalized to `fishery.catchability.fsh{idx}.sp{idx}`. |
| `cfg` has `species.liinf.sp0` (single-char typo) | Steps 1-3 fail. Normalized to `species.liinf.sp{idx}`. Suggestion via difflib against `known.patterns`: returns `species.linf.sp{idx}` (ratio ≈ 0.98). Message: `"Unknown config key 'species.liinf.sp0' — did you mean 'species.linf.sp{idx}'?"` |
| `cfg` has `species.totallywrong.sp0` | Steps 1-3 fail. Normalized form has no close match (cutoff 0.85). Message: `"Unknown config key 'species.totallywrong.sp0'"` (no suggestion clause). |
| `cfg` value is not a string (e.g. numeric int/float pre-parse) | Validate the KEY ignoring the value. `validate()` iterates `cfg.keys()`, not `cfg.items()`, so value type is irrelevant. |
| Mode = `error` and ≥ 1 unknown | Walk ALL keys, collect every `UnknownKey`, render one multi-line message, raise a single `ValueError`. Not fail-fast — users see the full list on first failure. |
| Mode = `off` and ≥ 1 unknown | One info-level log via `logging.getLogger("osmose.config")`: `"Config has N unknown keys; set validation.strict.enabled=warn for details."` Count only, no key names. |
| Mode = `off` and 0 unknowns | Silent. Zero log entries. |
| Mode = invalid string (`"verbose"`, `"true"`, `"ON"`, `"enabled"`, …) | `validate()` raises `ValueError("validation.strict.enabled must be one of 'off'/'warn'/'error'; got '<value>'")` on entry. Comparison is case-sensitive — lowercase only. Typo in the flag value itself is a startup bug, not a soft warning. |
| Flag absent from cfg | Defaults to `"off"` via `cfg.get("validation.strict.enabled", "off")` at the top of `EngineConfig.from_dict()`. |
| Validator runs before the reader's existing `KeyError` on missing required keys | By construction — the hook is at the very top of `from_dict` (before any `_get(cfg, "required.key")` call). If the user's cfg has BOTH unknown keys AND is missing required keys: `validate()` runs first, emits unknown warnings (or raises in `error` mode); only then does the reader's `KeyError` fire (in `warn`/`off` mode) on the first missing required key. |
| Pre-parsed flag and validator self-reference | `validation.strict.enabled` is registered via `osmose/schema/simulation.py`, so it's in `known.literals`. The validator never flags its own flag key as unknown. |

## Difflib tuning

**Key insight: normalize user's concrete key to pattern form BEFORE difflib.** Both sides of the comparison are then pattern strings (`species.liinf.sp{idx}` vs `species.linf.sp{idx}`), so `{idx}` bits cancel out in the ratio. Without this normalization, comparing the user's concrete `species.liinf.sp0` against the pattern `species.linf.sp{idx}` gives a ratio ≈ 0.81 — AT the naïve 0.8 cutoff, so single-character typos may or may not fire a suggestion depending on which character was flipped. That's brittle.

Call shape: `difflib.get_close_matches(_normalize_key_to_pattern(user_key), known.patterns, n=1, cutoff=0.85)`.

- `n=1`: single suggestion only (per brainstorm).
- `cutoff=0.85`: ratcheted conservative AFTER normalization. Normalized `species.liinf.sp{idx}` vs `species.linf.sp{idx}` ratio ≈ 0.98 — well above cutoff. Normalized `species.totallywrong.sp{idx}` vs anything ≈ 0.45 — no suggestion. Boundary case: `species.linfe.sp{idx}` (one trailing `e`) vs `species.linf.sp{idx}` ratio ≈ 0.95 — fires, correctly.
- **Suggestion rendering:** the warning message shows the pattern form verbatim (`did you mean 'species.linf.sp{idx}'?`) rather than a resolved concrete form (`did you mean 'species.linf.sp0'?`). Rationale: the pattern is honest about what the schema accepts (any species index). A concrete suggestion would mislead when the user typed `species.liinf.sp5` — we'd falsely point at `species.linf.sp0` as if the `sp0` part were relevant.

**Normalization details for `_normalize_key_to_pattern(user_key)`:** apply these substitutions in order against regex-anchored trailing segments:
- `sp\d+` → `sp{idx}`
- `fsh\d+` → `fsh{idx}`
- `map\d+` → `map{idx}`
- `age\d+` → `age{idx}`
- `sz\d+` → `sz{idx}`

Applied to every segment of the dot-separated key (not just the last), so `fishery.catchability.fsh0.sp3` → `fishery.catchability.fsh{idx}.sp{idx}` — matches multi-index patterns cleanly. For keys without any recognized trailing index, normalization is a no-op (the key is its own pattern form).

## Log destination

All output goes through `osmose.logging.setup_logging("osmose.config")`. Existing log routing applies. Users filtering on the `osmose.config` logger name can silence/capture validator output without touching Python's global `warnings` machinery.

The `=error` mode raises; does not log (raising is already visible).

## Non-goals

Explicit exclusions so a future reader or plan reviewer doesn't think they were missed:

- **Missing-required-key checks.** A separate validator for "config is missing `species.linf.sp0` but has `simulation.nspecies=1`" is a follow-up. Today the reader raises `KeyError` on the first missing required species key (`osmose/engine/config.py:45`), which is loud enough for the v0.9.2 cut.
- **Value-level validation beyond ParameterRegistry.** `field.validate_value()` already checks bounds/choices/types for schema-registered fields; we don't re-implement that.
- **Auto-fix / mutation of user's config dict.** The validator is read-only.
- **IDE / LSP integration.** Separate project.
- **Per-module validation profiles** (e.g. "genetics keys allowed only when genetics_enabled"). All keys share one union; a spurious `evolution.trait.imax.mean.sp0` in a non-genetics config is NOT flagged — it's a reader-known key that the reader will simply ignore when `genetics_enabled=False`. Adding conditional-module validation is future work.
- **Error recovery.** In `error` mode, startup fails. No "collect warnings and continue." If users want soft behavior, they use `warn`.

## Testing strategy

**Log-capture mechanism (pinned):** all tests that assert log output use pytest's built-in `caplog` fixture against the `osmose.config` logger. `osmose.logging.setup_logging` is a thin wrapper over stdlib `logging.getLogger(name)` (see `osmose/logging.py:7`), so `caplog.at_level(logging.INFO, logger="osmose.config")` works uniformly. No `capfd`, no monkeypatching, no custom handler.

**Unit tests** for `config_validation.py` (new file `tests/test_config_validation.py`):

1. `test_known_key_no_warning` — cfg with only `species.linf.sp0`; `validate(cfg, "warn")` returns empty list, `caplog.records` empty.
2. `test_unknown_key_warns_in_warn_mode` — cfg with `species.liinf.sp0`; mode=`warn`; asserts one log entry whose message contains `"species.linf.sp{idx}"` (the pattern-form suggestion).
3. `test_unknown_key_raises_in_error_mode_single` — same single unknown; mode=`error`; asserts `ValueError` raised and message contains the unknown key + the suggestion.
4. `test_error_mode_collects_all_unknowns` — cfg with TWO unknowns (`species.liinf.sp0` AND `predation.zzzbogus`); mode=`error`; asserts one `ValueError` whose message mentions BOTH keys. Verifies the collect-all-before-raise contract (a fail-fast impl would pass test #3 but fail this one).
5. `test_unknown_key_nudge_in_off_mode` — cfg with 1+ unknowns; mode=`off`; asserts exactly ONE log entry matching regex `r"Config has \d+ unknown keys"` and zero per-key entries.
6. `test_off_mode_zero_unknowns_silent` — cfg with only known keys; mode=`off`; asserts `caplog.records == []`. Confirms "silent on clean configs" claim.
7. `test_close_match_suggestion_cutoff` — cfg with `species.totallywrong.sp0`; mode=`warn`; asserts log entry exists AND does not contain `"did you mean"`.
8. `test_idx_pattern_accepts_concrete_key` — cfg with `species.linf.sp47`; mode=`warn`; no warning emitted.
9. `test_double_index_pattern_accepts_concrete_key` — cfg with `fishery.catchability.fsh0.sp3` (two indices in one key); mode=`warn`; no warning emitted. Verifies multi-index normalization (`fsh{idx}.sp{idx}`).
10. `test_internal_marker_not_warned` — cfg with `_osmose.config.dir`; mode=`warn`; no warning.
11. `test_non_string_value_still_validates_key` — cfg with key `species.linf.sp0` whose value is a `float` rather than a `str`; mode=`warn`; no warning for the key. Validates "skip value, check key" semantics.
12. `test_invalid_mode_raises` — mode=`"verbose"`; `validate(cfg, "verbose")` raises `ValueError`. Also checks case sensitivity: mode=`"OFF"` raises (not accepted).
13. `test_ast_extracts_cfg_get_literals_from_fixture` — unit test `_extract_literal_keys_from_config_py` against a synthetic small AST fixture containing `cfg.get("x.y")`, `cfg["a.b"]`, `_enabled(cfg, "p.q")`, and `_species_float(cfg, "sp{i}", n)`. Asserts normalized set `{"x.y", "a.b", "p.q", "sp{idx}"}`.
14. `test_ast_extracts_from_real_config_py_canary` — run `_extract_literal_keys_from_config_py()` on the real `osmose/engine/config.py` and assert the result contains these known-present sentinels: `_osmose.config.dir`, `simulation.nspecies`, `simulation.time.ndtperyear`. Cheap canary against walker drift.
15. `test_ast_handles_fstring_with_name_interpolation` — synthetic fixture with `cfg.get(f"mortality.{variant}.sp{i}")`; asserts captured as `"mortality.{idx}.sp{idx}"`.
16. `test_ast_drops_fstring_with_complex_interpolation` — synthetic fixture with `cfg.get(f"x.{obj.attr}")`; asserts it is NOT captured (can't resolve statically).
17. `test_ast_handles_in_comparison` — synthetic fixture with `"bioen.enabled" in cfg`; asserts `"bioen.enabled"` captured.
18. `test_source_file_unavailable_falls_back_to_schema_only` — monkeypatch `importlib.resources.files(...)` to raise `FileNotFoundError`; assert `build_known_keys()` returns a `KnownKeys` populated from `ParameterRegistry` only (schema 220 fields), with an info-level log about degraded mode; `validate()` continues to work.

**Integration tests** — extend the existing `tests/test_engine_config_validation.py` (which today tests `EngineConfig.__post_init__` field-level validation; unknown-key validation at `from_dict` entry is semantically adjacent):

19. `test_from_dict_off_mode_silent_on_example_configs` — **parametrized** over `[("eec", "data/examples/eec/"), ("bay_of_biscay", "data/examples/bay_of_biscay/"), ("baltic", "data/baltic/")]`. For each, load the all-parameters.csv and assert no WARNING/ERROR log entries, no exception. Catches AST-walker gaps specific to any one example config (e.g., a Baltic-only reader helper).
20. `test_from_dict_warn_mode_catches_known_typo` — load EEC config + inject `species.liinf.sp0`, mode=`warn`; assert warning logged with `species.linf.sp{idx}` suggestion.
21. `test_from_dict_error_mode_raises_with_typo` — same injection, mode=`error`; assert `ValueError` raised.

**Baseline parity:** full suite at HEAD `e8294ee` is 2485 passed / 15 skipped / 41 deselected (2500 collected). Target after implementation: **2485 + 21 = 2506 passed** (test set expanded from the original 13 to 21 per iteration-1 review findings). Any regression of a pre-existing test means the new hook is interfering with something — investigate rather than working around.

**Baseline parity test**: the full suite (currently 2485 passing at HEAD `d124de6`) must remain at `baseline + 13` after implementation — i.e. no pre-existing test regresses because its config happens to contain unknown keys that the new validator now flags.

Any pre-existing test config that carries an unknown key is a latent bug the validator exposes. In that case we fix the test config (or allowlist the key if it's a legitimate reader-honored pattern the AST walker missed).

## Rollout

Ship as v0.9.2 alongside any other cleanups. Default mode is `off`, so **zero user-visible change on clean configs**. Any pre-existing example config carrying an unknown key will now emit one new `osmose.config`-logger info line per load (`"Config has N unknown keys; set validation.strict.enabled=warn for details."`) — a necessary nudge to surface latent typos, but not a warning or error. CI test matrix should confirm zero unknowns on `data/examples/eec/`, `data/examples/bay_of_biscay/`, and `data/baltic/` configs before shipping; any legit reader-honored key the AST walker misses gets added to a module-local supplementary allowlist.

Users who want typo-catching flip `validation.strict.enabled=warn` in their config. Once the allowlist has been exercised against all three reference configs and stays stable across one or two releases, a future version can flip the default to `warn`.

## References

- Roadmap: `docs/parity-roadmap.md` Phase 7.3.
- Current config reader: `osmose/engine/config.py` (1894 lines).
- ParameterRegistry: `osmose/schema/registry.py`.
- Current `EngineConfig.from_dict`: `osmose/engine/config.py:1348`.
- Schema builder: `osmose/schema/__init__.py::build_registry()`.
- Logging helper: `osmose.logging.setup_logging`.
- Python stdlib: `ast`, `difflib.get_close_matches`, `functools.cache`.
