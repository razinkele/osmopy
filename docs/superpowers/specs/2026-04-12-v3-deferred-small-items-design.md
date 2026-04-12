# D-1 + M-5 + M-9: Deferred Small Items Bundle — Design Spec

> Three independent items from deep review v3, bundled into one plan because each
> is too small for a standalone spec/plan cycle.

---

## D-1: Reactive Isolate Write-Propagation Test

### Problem

The v3 architecture reviewer questioned whether `state.dirty.set(True)` inside
`reactive.isolate()` (e.g., `ui/pages/forcing.py:136-138`) actually propagates to
downstream readers. Investigation confirms it does — `reactive.isolate()` only
suppresses *reads* from creating dependencies; writes always propagate. But this
non-obvious behavior needs a regression test to prevent future reviewers from
"fixing" correct code.

### Solution

Add `test_reactive_write_inside_isolate_propagates` in `tests/ui/test_ui_reactive.py`.
The test creates a minimal Shiny reactive graph:
- A `reactive.Value` (the "dirty flag")
- An `@reactive.effect` that writes to it inside `reactive.isolate()`
- A downstream reader that observes the value

Assert the downstream reader sees the write. This pins the Shiny semantics contract.

### Deliverable

1 new test file, 1 test, ~25 lines.

---

## M-5: Java Source Investigation for Per-Species Seeding Year Max

### Problem

`population.seeding.year.max` is parsed as a global key and broadcast to all species
via `np.full(n_sp, ...)` in `config.py:901-906`. Java OSMOSE may support a per-species
variant `population.seeding.year.max.sp{i}`. If it does, Python silently ignores those
keys.

### Solution

**Phase 1 — Investigation:**
Search the Java OSMOSE GitHub repo (https://github.com/osmose-model/osmose) for
`population.seeding.year.max`. Determine if the Java parser reads per-species keys.

**Phase 2 — Depending on outcome:**
- **If Java supports per-species:** Add per-species parsing using the existing
  `_species_float_optional` pattern (like `population.seeding.biomass.sp{i}`).
  Fall back to the global key when per-species is absent. Add test.
- **If Java is global-only:** Add a comment in `config.py:901` documenting that
  global-only is correct Java parity. Close the item.

### Deliverable

Either a code change + test, or a documentation-only commit.

---

## M-9: UI Helper Extraction (4 Pages)

### Problem

5 UI page files were flagged for having zero pure helpers — all logic lives inside
reactive handlers, making it untestable. `economic.py` is a 47-line stub with no logic,
so the real scope is 4 pages.

### Approach

For each page: identify 1-2 data-transformation blocks inside reactive handlers,
extract to module-level pure functions, test in `tests/ui/test_ui_<page>.py`.

**Cross-file shared helper:** `parse_nspecies(cfg)` is duplicated in `movement.py` (2x)
and `forcing.py` (1x). Extract to `ui/pages/_helpers.py`.

### Per-page extraction targets

**movement.py (120 lines):**
- `parse_nspecies(cfg: dict) -> int` — extract from lines 51, 87 (duplicated)
- `count_map_entries(cfg: dict) -> int` — extract regex-based map counting from
  `sync_n_maps_from_config()` lines 111-117

**fishing.py (99 lines):**
- `collect_resolved_keys(fields, count) -> list[str]` — extract key resolution
  loop duplicated in `sync_fishery_inputs` and `sync_mpa_inputs` (lines 87-88, 96-97)

**forcing.py (139 lines):**
- `build_resource_updates(fields, cfg, input_getter, n_resource) -> dict` — extract
  the update-collection loop from `sync_resource_inputs()` lines 115-131
- Uses `parse_nspecies()` from shared helper

**diagnostics.py (90 lines):**
- `format_timing_rows(timing: dict) -> list` — extract HTML table row building
  from `diag_timing()` lines 65-69

### Testing

Each extracted helper gets 2-3 tests in `tests/ui/test_ui_<page>.py`:
- Happy path with typical input
- Edge case (empty input, zero count)
- One boundary condition specific to the helper

### What stays reactive

State inspection, input reading, `reactive.isolate()` blocks, state mutations, and
UI update calls remain inside handlers — they depend on Shiny machinery.

### Deliverable

- 1 new shared module: `ui/pages/_helpers.py`
- 4 modified page files (movement, fishing, forcing, diagnostics)
- 4 new test files in `tests/ui/`
- ~10-15 new tests total
