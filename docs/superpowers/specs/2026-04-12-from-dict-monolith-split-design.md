# I-3: EngineConfig.from_dict Monolith Split — Design Spec

> Deep review v3 finding I-3. Deferred from the remediation plan because the refactor
> touches 611 lines with high regression risk and needs per-subsystem tasks.

## Problem

`EngineConfig.from_dict()` in `osmose/engine/config.py` (lines 824-1435) is a 611-line
method that parses 11+ subsystems inline. It already delegates to 12 helper functions,
but the remaining inline blocks are 40-136 lines each. This makes the method hard to
read, test in isolation, and maintain.

## Goal

Split `from_dict` into a coordinator that calls 5 subsystem parsers, each owning a
cohesive block of config parsing. No behavioral changes — pure refactor with parity
gate after each extraction.

## Approach

### What gets extracted

| Helper | Current lines | Size | What it parses |
|--------|--------------|------|----------------|
| `_parse_growth_params(cfg, n_sp, n_dt)` | 863-920 | ~57 | VB params (linf, k, t0), egg size, allometry, lifespan, lmax, vb_threshold_age |
| `_parse_predation_params(cfg, n_sp, n_bg, species_names)` | 921-1057 | ~136 | Feeding stages, size ratios, ingestion rate, critical success, background feeding, 2D array padding |
| `_parse_reproduction_params(cfg, n_sp, n_dt)` | scattered ~40 | ~40 | Sex ratio, fecundity, maturity size/age, seeding biomass/duration, larva mortality |
| `_merge_focal_background(focal_arrays, bg_info, n_bg)` | 1085-1164 | ~79 | Concatenate focal-only arrays with background species zero-defaults |
| `_parse_output_flags(cfg, n_sp)` | 1321-1434 | ~113 | Distribution by-age/by-size, size bins, bioen output, record frequency, diet output |

### What stays inline

- Simulation basics (lines 825-828, 4 lines) — too small
- Movement params (lines 1069-1083, 15 lines) — too small
- Bioenergetics block (lines 1270-1310) — already uses utility helpers extensively
- Genetics/economics (lines 1312-1319, 5 lines each) — trivial
- Fishing features phase 2 (lines 1192-1216) — already delegates to helpers

### Return type for helpers

Each helper returns a plain `dict[str, Any]` of field-name → value pairs. The
coordinator unpacks them into the `EngineConfig(...)` constructor call. This avoids
introducing intermediate dataclasses and keeps the refactor minimal.

Alternative considered: typed `NamedTuple` per subsystem. Rejected because it adds 5
new types for a refactor that should minimize surface-area changes. Can revisit later.

### After the refactor

`from_dict` becomes a ~150-line coordinator:
1. Parse simulation basics (inline, 4 lines)
2. Call `_parse_growth_params()` → growth dict
3. Call `_parse_predation_params()` → predation dict
4. Call `_parse_reproduction_params()` → reproduction dict
5. Call existing helpers (_parse_fisheries, etc.)
6. Call `_merge_focal_background()` → merged arrays
7. Parse movement, bioen, genetics, economics (inline, small)
8. Call `_parse_output_flags()` → output dict
9. Assemble `EngineConfig(**{...all dicts merged...})`

## Risk mitigation

- Each extraction is a separate task with its own `pytest + ruff + parity` gate
- If any extraction breaks parity (12/12 bit-exact), revert that single task
- No new tests required — existing 2148 tests + 12 parity tests are the safety net
- Helper function signatures include only `cfg` dict + dimensions (n_sp, n_dt, etc.)
  to avoid coupling to EngineConfig internals

## Out of scope

- Splitting `EngineConfig` dataclass itself (80+ fields stay as-is)
- Changing the config key parsing logic
- Adding new validation (that's separate findings)
- Refactoring the return statement (it's the natural assembly point)
