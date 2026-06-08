# Real-time config validation in the Shiny form — Design

**Date:** 2026-06-08
**Status:** Approved direction (brainstormed; codebase-grounded; **2 in-loop review rounds folded**). Clean UI feature over existing, tested validators.

## Motivation

The Setup form renders inputs from the schema but gives **no validation feedback while editing** — type/range/enum errors, missing species names, and missing file references only surface when the user clicks **Run**, which blocks the run and dumps the reasons into the run log. The validators to catch all of this already exist and are tested (`validate_config`, `check_file_references`, `check_species_consistency`); they are simply never shown during editing. This adds a **live validation summary panel** on the Setup page that mirrors the Run gate as you type, so problems are visible (and fixable) before a run is attempted. It is a UI-integration feature — **no new validation logic**.

## Verified context (audit)

Read in the current tree:

- **No user-facing validation in the form today.** `ui/components/param_form.py::render_field` returns a bare input with **no error slot**; `render_category` is a flat `ui.div`; the species panel is a dense spreadsheet (`render_species_table`). Only float/int parse fallbacks are *logged*. So per-field inline errors would require restructuring the whole render path and are awkward in the table → out of scope (see Scope).
- **The Run gate is the exact logic we want** (`ui/pages/run.py::handle_run`, ~:476-484):
  ```python
  config = state.config.get()
  errors, warnings = validate_config(config, state.registry)
  source_dir = state.config_dir.get()
  if source_dir:
      errors.extend(check_file_references(config, str(source_dir), state.registry))
  warnings.extend(check_species_consistency(config))
  ```
- **Validators** (`osmose/config/validator.py`): `validate_config(config: dict, registry) -> tuple[list[str], list[str]]` (errors, warnings — type/range/enum, multi-value aware); `check_file_references(config, base_dir: str, registry) -> list[str]` (errors); `check_species_consistency(config) -> list[str]` (warnings). All return human-readable strings; none raise on a partial config.
- **State** (`ui/state.py::AppState`): `state.config: reactive.Value[dict[str,str]]` (updated on every edit by the Setup `sync_inputs` effects), `state.registry` (the schema registry), `state.config_dir: reactive.Value[Path|None]` (set only when a config was loaded from disk), `state.load_trigger`.
- **Setup page** (`ui/pages/setup.py`): `setup_ui()` is `ui.layout_columns(card(simulation_fields), card(species_panels), col_widths=[4, 8])`; `setup_server(input, output, session, state)` already holds the reactive `sync_simulation_inputs` / `sync_species_inputs` effects that write `state.config`.
- **reader `ConfigDiagnostic`** is PARSE-level (file I/O) — irrelevant to a live in-memory form; only meaningful in the Advanced raw-import tab → out of scope.

## Architecture

One pure helper — the single source of truth for "what's wrong with this config" — consumed by **all three** of its current callers (the new live panel, the existing Run gate, and the `osmose validate` CLI, which today hand-rolls the identical sequence at `osmose/cli.py:29-33`), plus the panel UI. Unifying the CLI too is what makes "single source of truth" actually true rather than aspirational (round-2 finding: `cli.py` is a third copy).

### 1. `summarize_config_validation` (`osmose/config/validator.py`)

```python
def summarize_config_validation(
    config: dict[str, str],
    registry: ParameterRegistry,
    config_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Aggregate all config checks into (errors, warnings) of human-readable strings.

    Mirrors the Run gate exactly: validate_config (type/range/enum) + file-reference
    checks (only when config_dir is set — relative paths would false-error otherwise) +
    species-name consistency (warnings). Pure; never raises on a partial config.
    """
    errors, warnings = validate_config(config, registry)
    if config_dir:  # truthiness — matches the gate's `if source_dir:` exactly (an empty
        errors.extend(check_file_references(config, str(config_dir), registry))  # str "" never occurs)
    warnings.extend(check_species_consistency(config))
    return errors, warnings
```
This is the **exact** current Run-gate sequence (run.py:478-485, which uses named `file_errors`/
`species_warnings` intermediates), extracted. Behavior-preserving by construction — verified by
prototype: helper == inline gate byte-identical on the real 601-key baltic config with
`config_dir` None / a real dir / an empty dir. (`validate_config` returns fresh lists and does NOT
itself call the other two checkers, so no double-counting. `validator.py` already imports `Path`
and `ParameterRegistry` — the hints need NO new imports.) The guard uses `if config_dir:`
truthiness, not `is not None`, so it matches the gate even for the (never-occurring) `""` case.

### 2. Caller refactors (`ui/pages/run.py::handle_run` + `osmose/cli.py::cmd_validate`) — DRY

Replace the inline 6-line sequence with:
```python
config = state.config.get()
errors, warnings = summarize_config_validation(config, state.registry, state.config_dir.get())
```
The downstream `if errors: ... if warnings: ...` blocking logic is **unchanged** (verified: `errors`/`warnings` are consumed only within the gate; `config = state.config.get()` stays — it's still passed to the engine runners). This keeps the panel and the gate from drifting. A test locks that the helper reproduces the old inline result. **Prune** the now-unused `validate_config`/`check_file_references`/`check_species_consistency` imports from `run.py` (replace with `summarize_config_validation`) or ruff F401 fails CI. (Round-2 verified: those three names appear in `run.py` **only** at the import and the gate block — `run.py:11-13` + `:479/482/484` — so the prune is safe; run.py imports from `osmose.config.validator` directly, so no `osmose/config/__init__.py` re-export is needed and none should be added.)

Likewise refactor the CLI `cmd_validate` (`osmose/cli.py:29-33`), which today inlines the same three calls with `config_dir = config_path.parent` (always set — a CLI always reads from a file, so the helper's `if config_dir:` guard always runs the file checks, exactly matching today):
```python
errors, warnings = summarize_config_validation(config, registry, config_path.parent)
```
Same import prune (`cli.py:18-22`). Behavior-preserving by construction (the helper *is* that sequence). The DRY-lock test covers both call sites. After this, the helper genuinely is the only copy.

### 3. Live panel (`ui/pages/setup.py`)

- `@reactive.calc _config_validation()` → reads `state.config.get()` (so it recomputes on every edit via the existing sync effects — confirmed: the Setup `sync_simulation_inputs`/`sync_species_inputs` effects are bare `@reactive.effect`s that `state.config.set(...)` on every input change; the calc only READS config, so no reactive loop) + `state.config_dir.get()`, calls `summarize_config_validation(...)`, and returns `(loaded: bool, errors, warnings)` where `loaded = bool(config) and "simulation.nspecies" in config`. Wrapped in a defensive `try/except Exception` that degrades to `(True, [], ["validation unavailable: <e>"])` — the Setup tab must never crash on a half-entered config.
- `@render.ui config_validation()` → a full-width card. Content (three states):
  - **No config** (`not config` or `"simulation.nspecies" not in config`): a neutral grey line `No configuration loaded` — **NOT** green. (Avoids the false "✓ valid" on an empty/unloaded `{}` config, which `validate_config({})` would otherwise report as clean.)
  - **Clean** (config present, no errors/warnings): a single quiet green line `✓ Configuration valid`.
  - **Issues:** a header badge `N error(s) · M warning(s)` (red when errors>0, else amber), then an errors group (red `•` lines) followed by a warnings group (amber `•` lines), each line the validator's own message string (which includes the offending key).
    - **Cap the volume (round-2 SHOULD-FIX — real on loaded configs).** `check_file_references` emits **one error per FILE_PATH key**, and those keys are heavily index-expanded — `movement.file.map{idx}` alone is ~26 keys on baltic / ~56 on eec_full, plus `species.file.sp{idx}`, `reproduction.season.file.sp{idx}`, etc. A single moved/renamed `config_dir` therefore produces **30-60+ red lines** that bury the handful of type/range errors the user is actively editing. So **cap each group at the first 10 lines** with a muted `… and N more error(s)` / `… and M more warning(s)` tail line. (The Run gate's run-log dump stays uncapped — this cap is panel-only, for at-a-glance scanning.)
    - **Wrap long messages (round-2 SHOULD-FIX).** A resolved `File not found for 'movement.file.map0': /abs/path/...csv` line is ~90-110 chars, and `.card` sets `overflow: hidden` (`www/osmose.css:463`) with **no** `word-break`/`overflow-wrap` rule on body text, so long paths clip silently. Give the message lines `style="overflow-wrap:anywhere"` (or a small `.osm-validation-msg` class) so they wrap instead of clip.
- **Accessibility (round-2 SHOULD-FIX).** Wrap the card's inner content `ui.div` with `**{"aria-live": "polite", "aria-atomic": "true"}` so the live summary is announced to screen readers as it updates (atomic → re-announce the whole summary, not diffed lines). In-repo precedent: the live calibration header at `ui/pages/calibration_handlers.py:837` uses exactly `aria-live="polite"`. Severity is already non-color-redundant via the `N error(s) · M warning(s)` text badge and the `✓` glyph; optionally use distinct per-line glyphs for errors vs warnings (e.g. `✕` red / `⚠` amber) so a colorblind user scanning lines — not just the badge — can tell them apart.
- `setup_ui()` returns `ui.div(expand_tab(...), ui.layout_columns(...), id="split_setup", class_="osm-split-layout ...")`. Insert `ui.output_ui("config_validation")` (wrapped in a compact `ui.card`) as the **second child of that outer `ui.div`** — after `expand_tab(...)`, before `ui.layout_columns(...)`. NB: the outer div is a flex/split-layout container, so the manual Playwright pass must confirm the card spans **full width above** the columns (not collapses into a flex column); if it does, give the card `class_="w-100"` or move it just outside the split container.

## Data flow

`edit field → sync_inputs effect writes state.config → _config_validation recomputes (summarize_config_validation) → config_validation re-renders the card`. The Run gate calls the same helper at click time. No new state, no engine calls.

**Confirmed (round-2): the panel also refreshes on config LOAD, not only on edit.** Every loader sets `state.config` *directly* — example load `ui/pages/grid.py:843`, scenario load `ui/pages/scenarios.py:181`, advanced import `ui/pages/advanced.py:175` — and that `reactive.Value` is exactly the calc's dependency, so the calc invalidates immediately on load (it does **not** need `load_trigger`, which only re-renders the input widgets). Equally confirmed: the panel is **inert during a run** (the engine runs off a copied `run_config`; `state.config` is not mutated, and the calc does not read `state.busy`), and editing a species-table cell writes config on the **same flush** (the `sync_species_inputs` effect re-reads all cells and does one `state.config.set()` per run, so no one-interaction lag).

**Coverage caveat (honest):** the panel validates exactly the keys in `state.config`. The Setup species sync only writes **visible, synced** fields — advanced species fields hidden behind "Show advanced parameters" and multi-value (`;`-array) fields are NOT written back from the form (they retain their loaded-from-disk value, or are absent for a fresh in-UI config). So the panel mirrors the Run gate on the *synced* key set, not necessarily on every on-screen-editable widget. This is inherent to the existing sync design, not introduced here — the panel just surfaces whatever `validate_config` sees on `state.config`. The manual run-through must toggle "Show advanced parameters" and confirm the panel does not show errors for keys with no visible/editable input.

## Error handling / edge cases

- **Partial config** (nspecies typed but species blank; a value mid-edit): the validators return strings, don't raise; the calc's `try/except` is a backstop so any unexpected error becomes a warning line rather than a broken tab.
- **No config dir**: `check_file_references` is skipped (passing `config_dir=None`), so a fresh in-UI config isn't flagged for relative/absent paths it legitimately hasn't resolved yet — matching the Run gate.
- **Perf**: `validate_config` over ~200 keys is sub-millisecond; inputs already debounce client-side. No explicit debounce (trivial follow-on if it ever feels chatty).
- **Empty / unloaded config**: `validate_config({})` returns no errors, which would falsely read as `✓ Configuration valid` for a non-runnable `{}`. The `loaded` flag handles this → the panel shows the neutral **"No configuration loaded"** state instead. (In practice the Setup tab's first reactive pass writes `simulation.nspecies=3` + visible species fields, so truly-empty is transient — but the neutral branch is correct for the unloaded/`{}` case regardless.)

## Testing

- **Unit-test the pure helper** `summarize_config_validation` (`tests/test_validator.py` or a new `tests/test_config_validation_summary.py`): a clean `sample_config` → `([], [])` (or warnings-only); an out-of-range value → the expected error string; a missing `species.name.spN` (nspecies>names) → the consistency warning; `config_dir=None` → file checks skipped (no file errors even with a FILE_PATH key); `config_dir=<tmp without the file>` → a file-reference error. A malformed value can't crash it (returns lists).
  Also test the `loaded` flag: empty `{}` and a config without `simulation.nspecies` → `loaded=False`; a config with it → `loaded=True`.
- **DRY-lock test**: assert `summarize_config_validation(cfg, reg, d)` equals the old inline composition (`validate_config` + conditional `check_file_references` + `check_species_consistency`) for a representative config — so the gate **and CLI** refactors are provably behavior-preserving. (Round-2 verified the result lists are deterministic — `validate_config`/`check_file_references`/`check_species_consistency` all iterate `config.items()`/`range(...)` with no glob or set ordering, and all return **fresh** local lists, so the helper's `.extend()` can't alias/mutate a cached list — direct list `==` is sound, no sorting needed, no repeat-call test required.)
- **Wiring**: assert `config_validation` and `summarize_config_validation` appear in `ui/pages/setup.py` source; assert `summarize_config_validation` is used in `ui/pages/run.py` **and `osmose/cli.py`** AND that the old `validate_config`/`check_file_references`/`check_species_consistency` imports were pruned from both (so ruff F401 stays clean). `validator.py` has no `__all__`, so these stay source-string asserts (correct mechanism). Page-build smoke `tests/test_ui_*` (esp. any Setup/run UI build test) still passes; CLI `cmd_validate` behavior test (existing `tests/test_cli*` if any, else add one) still passes; `ruff check ui/ osmose/ tests/` + `ruff format --check` clean.
- **Manual Playwright run-through** (render fn not unit-tested, per convention): load a config, edit a numeric field out of range → the card shows the error live; fix it → card returns to green; before any load → neutral "No configuration loaded"; toggle "Show advanced parameters" → no errors appear for keys with no visible input; confirm the card spans full width above the columns (placement) and that rapid species-table typing doesn't cause objectionable flicker (the panel re-renders independently of the inputs, so no focus loss — but watch the cadence); the existing Run gate still blocks on the same errors. **Round-2 additions:** load a config then point it at a moved/renamed dir (or load one with broken file refs) → confirm the errors group **caps at 10** with a `… and N more` tail (not a 50-line wall) and that a long File-not-found **path wraps, not clips**; confirm the card carries `aria-live="polite"`.

## Scope / YAGNI

- **In:** the `summarize_config_validation` helper, the Run-gate **and CLI** refactors to use it, the Setup live panel (calc + render + `output_ui`, with the 10-line cap, message wrapping, and `aria-live`), tests, a CHANGELOG note.
- **Out (clean follow-ons):** per-field inline error styling / red borders (Approach C — needs render-path restructure + awkward in the species table); surfacing the engine key-allowlist `validate()` unknown-key suggestions (engine-side, noisier — a separate "unknown key" advisory); the reader `ConfigDiagnostic` parse panel (only meaningful in the Advanced raw-import tab); blocking/auto-fix behavior (the panel is informational; the Run gate keeps its blocking authority unchanged).

## Honest limitations

- The panel is a **summary**, not pinpoint per-field highlighting — it lists `key: message` lines, not red borders on the offending inputs (that's the deferred Approach C).
- It mirrors the Run gate's checks exactly; it does **not** add new validation (e.g. unknown-key detection is engine-side and out of scope).
- It validates the **synced** key set in `state.config`, which excludes advanced-hidden species fields and multi-value `;`-array fields that the form doesn't write back (Coverage caveat above) — same limitation the Run gate already has, since both read `state.config`.
- **`config_dir` can be stale across loads (round-2 note).** Only the example loader sets `state.config_dir` (`grid.py:844`); scenario load (`scenarios.py:179-205`) and advanced import (`advanced.py:165-194`) set `state.config` but leave `config_dir` untouched. So loading an example then a scenario can leave the panel's `check_file_references` resolving paths against the prior dir. This is **not introduced here** — `run.py` reads the same stale `state.config_dir.get()`, so the panel still mirrors the gate exactly (the DRY-parity invariant holds). Fixing the stale dir is a separate concern for both call sites.

## Delivery

Single additive PR: `osmose/config/validator.py` (+helper), `ui/pages/setup.py` (panel + output_ui), `ui/pages/run.py` (DRY refactor), `osmose/cli.py` (DRY refactor), tests, CHANGELOG. No schema/engine changes; the Run gate's blocking behavior is unchanged. CHANGELOG: a `**ui (config validation):**` line under `## [Unreleased]` → `### Added` (matches the repo's Keep-a-Changelog format).
