# Config Validation — Design

> **Roadmap item:** `docs/parity-roadmap.md` Phase 7.3. Spec for the plan at (forthcoming) `docs/superpowers/plans/2026-04-19-config-validation-plan.md`.

**Goal.** Catch OSMOSE config typos at load time instead of letting them produce silent downstream garbage. Today a misspelled key (`species.liinf.sp0` with an extra `i`) is silently ignored by the reader; the simulation runs with the default value and the user only finds out when the numbers look wrong. This spec adds a validator that flags unknown keys with optional typo suggestions and a mode flag for rollout safety.

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
    suggestion: str | None   # populated only when difflib confidence ≥ 0.8

def build_known_keys() -> frozenset[re.Pattern]:
    """Union of (ParameterRegistry field patterns) + (AST-extracted reader keys).

    Each entry is a compiled regex:
    - literal keys become ``re.escape(key)``
    - {idx}-pattern keys become ``re.escape(pattern).replace(r'\\{idx\\}', r'\\d+')``

    Cached at module import time via ``@functools.cache`` on a helper.
    """

def validate(cfg: dict[str, str], mode: str) -> list[UnknownKey]:
    """Return the list of unknown keys (empty = all known).

    ``mode`` controls side effects:
      - "off"   → return list, emit single-line summary if nonempty
      - "warn"  → return list, log each + suggestion via osmose.logging
      - "error" → if list nonempty, raise ValueError with full rendered message

    Raises ValueError for invalid mode strings.
    """
```

### AST walker

One module-local function `_extract_literal_keys_from_config_py() -> set[str]`:

- Parse `osmose/engine/config.py` with `ast.parse(pathlib.Path(__file__).with_name("config.py").read_text())`.
- Walk every `ast.Call` and `ast.Subscript` node.
- Extract string literals from:
  - `cfg.get("x", ...)` → first arg
  - `cfg["x"]` → subscript slice
  - `_enabled(cfg, "x")` → second arg (named `key`)
  - `_species_float(cfg, "species.x.sp{i}", n)` / `_species_int(...)` / `_species_array(...)` / `_species_string(...)` / `_species_bool(...)` / `_get(cfg, "x")` → second arg
- Keys containing `{i}`, `{fsh}`, `{idx}` are normalized to `{idx}` to match the `ParameterRegistry` convention.
- Result cached **lazily on first call** via `@functools.cache`, not at module import — avoids import-order fragility with `ParameterRegistry` submodule registration. The first `EngineConfig.from_dict()` call pays a one-time AST-parse cost (~10ms); subsequent calls are free.

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
    choices=("off", "warn", "error"),
    default="off",
    category="simulation",
    description=(
        "Unknown-config-key validation mode. 'off' (default): silent with "
        "a one-line summary if any unknowns found. 'warn': log each unknown "
        "key with a typo suggestion. 'error': raise ValueError listing all "
        "unknown keys."
    ),
),
```

This makes the validator recognize its own flag (avoids a self-referential "unknown key" warning for `validation.strict.enabled`).

## Failure modes and semantics

| Situation | Behavior |
|---|---|
| `cfg` contains `_osmose.config.dir` (internal marker) | Allowlisted — AST walker captures it (used literally in `config.py`) |
| `cfg` has a `{idx}`-concrete key like `species.linf.sp47` | Regex match against the `species.linf.sp{idx}` → `species\.linf\.sp\d+` pattern succeeds. Known. |
| `cfg` has `species.liinf.sp0` (typo) | Unknown. `difflib.get_close_matches("species.liinf.sp0", patterns, n=1, cutoff=0.8)` returns `["species.linf.sp{idx}"]`. Warning includes the suggestion. |
| `cfg` has `species.totallywrong.sp0` | Unknown. No close match (cutoff 0.8). Warning printed without suggestion. |
| `cfg` value is not a string (shouldn't happen but defensively) | Skip value; key is still validated. |
| Mode = `error` and ≥ 1 unknown | Collect ALL unknowns, render one message, raise single `ValueError`. Not fail-fast — users see the full list on the first failure. |
| Mode = `off` and ≥ 1 unknown | Single line via `osmose.logging.setup_logging("osmose.config")`: `"Config has 3 unknown keys; set validation.strict.enabled=warn for details."` Count only, no key names. |
| Mode = `off` and 0 unknowns | Silent. |
| Mode = invalid string (e.g. `"verbose"`) | Raise `ValueError` on entry to `validate()` — typo in the flag value itself is a startup bug, not a soft warning. |

## Difflib tuning

`difflib.get_close_matches(key, patterns, n=1, cutoff=0.8)`:
- `n=1`: single suggestion only (per brainstorm — top-3 adds noise)
- `cutoff=0.8`: ratcheted conservative. `species.linf.sp0` vs `species.liinf.sp0` ratio ≈ 0.95 — suggestion fires. `species.linf.sp0` vs `species.totallywrong.sp0` ratio ≈ 0.4 — no suggestion, plain warning only.
- Matches are compared against the *pattern* strings (`species.linf.sp{idx}`) not concrete keys, so the allowlist stays small (~312 entries rather than one per species).
- **Suggestion rendering:** the warning message shows the pattern form verbatim (`did you mean 'species.linf.sp{idx}'?`) rather than a resolved concrete form (`did you mean 'species.linf.sp0'?`). Rationale: the pattern is honest about what the schema accepts (any species index). A concrete suggestion would mislead when the user typed `species.liinf.sp5` — we'd falsely-point at `species.linf.sp0` as if the `sp0` part were relevant.

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

**Unit tests** for `config_validation.py` (new file `tests/test_config_validation.py`):

1. `test_known_key_no_warning` — `species.linf.sp0` in cfg; `validate(cfg, "warn")` returns empty list, no log entries.
2. `test_unknown_key_warns_in_warn_mode` — `species.liinf.sp0`; mode=`warn`; asserts one log entry containing `"species.linf.sp{idx}"` as suggestion.
3. `test_unknown_key_raises_in_error_mode` — same unknown key; mode=`error`; asserts `ValueError` raised, message contains both the unknown key and the suggestion.
4. `test_unknown_key_silent_in_off_mode` — same unknown key; mode=`off`; asserts exactly ONE log entry matching the summary regex `r"Config has \d+ unknown keys"` and no per-key entries.
5. `test_close_match_suggestion_cutoff` — `species.totallywrong.sp0`; mode=`warn`; asserts log entry exists AND does not contain the substring `"did you mean"`.
6. `test_idx_pattern_accepts_concrete_key` — `species.linf.sp47` in cfg; mode=`warn`; no warning emitted (regex-matches the pattern).
7. `test_internal_marker_not_warned` — `_osmose.config.dir` in cfg; mode=`warn`; no warning.
8. `test_invalid_mode_raises` — mode=`"verbose"`; `validate(cfg, "verbose")` raises `ValueError`.
9. `test_ast_parse_captures_cfg_get_literals` — unit test the `_extract_literal_keys_from_config_py` helper directly against a small test AST fixture (don't rely on `config.py` content that may drift).
10. `test_ast_parse_normalizes_idx_variants` — `cfg.get("species.k.sp{i}")` is captured as `species.k.sp{idx}` (matches the ParameterRegistry convention).

**Integration tests** — extend the existing `tests/test_engine_config_validation.py` (which today tests `EngineConfig.__post_init__` field-level validation; unknown-key validation at `from_dict` entry is semantically adjacent). Unit tests for the pure validator helpers go in a new `tests/test_config_validation.py` file (items 1-10 above).

11. `test_from_dict_off_mode_silent_on_valid_config` — load `data/examples/eec/` config; no warning, no raise.
12. `test_from_dict_warn_mode_catches_known_typo` — inject `species.liinf.sp0` into an EEC config, mode=warn; assert warning logged.
13. `test_from_dict_error_mode_raises_with_typo` — same injection, mode=error; assert `ValueError` raised.

**Baseline parity test**: the full suite (currently 2485 passing at HEAD `d124de6`) must remain at `baseline + 13` after implementation — i.e. no pre-existing test regresses because its config happens to contain unknown keys that the new validator now flags.

Any pre-existing test config that carries an unknown key is a latent bug the validator exposes. In that case we fix the test config (or allowlist the key if it's a legitimate reader-honored pattern the AST walker missed).

## Rollout

Ship as v0.9.2 alongside any other cleanups. Default mode is `off`, so zero user-visible change. Users who want typo-catching flip the flag in their config. Once the allowlist has been exercised against EEC / Bay-of-Biscay / Baltic example configs and stays stable for one or two releases, a future version can flip the default to `warn`.

## References

- Roadmap: `docs/parity-roadmap.md` Phase 7.3.
- Current config reader: `osmose/engine/config.py` (1894 lines).
- ParameterRegistry: `osmose/schema/registry.py`.
- Current `EngineConfig.from_dict`: `osmose/engine/config.py:1348`.
- Schema builder: `osmose/schema/__init__.py::build_registry()`.
- Logging helper: `osmose.logging.setup_logging`.
- Python stdlib: `ast`, `difflib.get_close_matches`, `functools.cache`.
