---
name: Phase 7.3 config-validation implementation (SHIPPED)
description: How EngineConfig.from_dict catches unknown keys — allowlist sources, three modes, and the one gotcha that matters for future schema/engine changes.
type: project
originSessionId: 62ae1657-034b-4171-9e07-85306c7671a8
---
**Shipped:** 2026-04-19 (master `327a20c..3af9f9a`), staged for v0.9.2 under `[Unreleased]`.

**What it does.** `EngineConfig.from_dict(cfg)` now validates every cfg key against an allowlist BEFORE any `_get(cfg, "simulation.nspecies")` lookup. Default mode is silent on clean configs. Opt-in via `validation.strict.enabled` config key — values `off` / `warn` / `error`.

**Allowlist sources (3-way union, ~391 total patterns):**
1. `ParameterRegistry.all_fields()` — 220 schema fields (221 including the new `validation.strict.enabled` flag itself).
2. AST walk of `osmose/engine/config.py` — captures all `cfg.get(literal)`, `cfg[literal]`, helper calls (`_get` / `_enabled` / `_species_float[_optional]` / `_species_int[_optional]` / `_species_str`), `"literal" in cfg` membership tests, Assign-RHS f-strings (the 8 `*_key = f"..."` fishery/mortality patterns), and f-strings with `Name` interpolation. The `.format()` shape is also captured.
3. `_SUPPLEMENTARY_ALLOWLIST` (40 entries in `osmose/engine/config_validation.py`) — reader-injected metadata (`osmose.configuration.*`, `osmose.version`) + legacy aliases (`species.lw.*`) + Java-side keys (`movement.*.map{idx}`, `output.diet.stage.*`, `simulation.restart.enabled`, etc.).

**Match order (fast → slow):** literal exact match → normalized-pattern match (`sp\d+` → `sp{idx}` segment-by-segment) → compiled regex fallback → `difflib.get_close_matches(n=1, cutoff=0.85)` for suggestions.

**The one gotcha.** When adding a new config key the engine reads:
- If config.py reads it via one of the captured AST shapes → walker captures it automatically, nothing to do.
- If the key is built dynamically (e.g., from a caller-arg `key_pattern` two frames up, as in `_load_per_species_timeseries`) → add pattern to `_SUPPLEMENTARY_ALLOWLIST` (prefer `{idx}` form over enumerated literals).
- If the key is pure reader-injected metadata or a legacy alias → add to `_SUPPLEMENTARY_ALLOWLIST`.
- If it's a real new parameter → register in `osmose/schema/*.py`, which auto-populates the allowlist via the first source.

**The integration test that guards the warning-free state:** `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs[eec|baltic|eec_full]`. If this breaks after adding a key, run the test and either extend the walker or allowlist the offender.

**Deliberate spec divergences (recorded in code comments):**
- Cache is a module-level `_KNOWN_KEYS_CACHE` dict keyed by `"full"`, NOT `@functools.cache`. Reason: degraded (schema-only) fallback on FS error must not be memoized.
- Logger uses `setup_logging("osmose.config")` (not bare `logging.getLogger`) to match `osmose/config/reader.py:10` — the osmose.config logger is shared.

**Key files:**
- `osmose/engine/config_validation.py` — the module
- `osmose/schema/simulation.py` — schema entry for `validation.strict.enabled` (ENUM, advanced, default `"off"`)
- `osmose/engine/config.py:1348` — 4-line hook at top of `from_dict`
- `tests/test_config_validation.py` — 20 unit tests
- `tests/test_engine_config_validation.py` (appended) — 3 integration tests (5 collected cases)
