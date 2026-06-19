# Scenario Wizard — Design

**Date:** 2026-06-19
**Status:** Approved (brainstorming complete)

## Goal

A guided **"New Scenario"** wizard that lets a user bootstrap a working scenario from a known-good starting point — a bundled demo (Baltic / Bay of Biscay / EEC / EEC Full / minimal) **or** a previously-saved scenario — set a few headline parameters up front, name it, and land in the normal editor with the new scenario already **persisted** and loaded. It closes the gap between today's "Load example" (loads a throwaway temp config in place, no clone/name/save path) and "Scenarios → Fork" (creates a separate entry but no guided edit), giving newcomers a smooth clone-and-customize flow.

## User flow

A **"+ New Scenario"** button on the existing **Scenarios** page opens a modal 3-step stepper:

1. **Preset** — choose a starting point: a bundled demo (from `list_demos()`) or a saved scenario (from `ScenarioManager.list_scenarios()`), shown as two grouped sections in one select.
2. **Basics** — set headline parameters:
   - **Years** → `simulation.time.nyear`
   - **Steps/year** → `simulation.time.ndtperyear`
   - **Reproducible RNG** (toggle) → sets both `movement.randomseed.fixed` and `stochastic.mortality.randomseed.fixed`
   Each is pre-filled from the chosen source's current value.
3. **Name** — the new scenario name (validated: non-empty, filesystem-safe, not a duplicate).

**Create** → builds the config (source + basics overrides) → **saves** it as a scenario via `ScenarioManager.save` (auto-save, so it appears in the list immediately) → loads it into the editor (`state.load_config` + `config_dir`/`config_name` + `load_trigger`) → closes the modal. The user is now in the editor on a named, persisted scenario, free to customize further and re-save.

## Scope

**In scope:**
- Pure, tested core that resolves a source to a config, applies the three headline overrides, validates a name, and assembles the result.
- A modal stepper added to the existing Scenarios page (no new nav tab → no `#main_nav` visual-baseline change).
- Auto-save-and-load on Create.

**Out of scope (YAGNI):**
- Seed-*value* persistence. The engine seed is a run-time argument (`run.py:294 seed=0`), not a config key; baking a seed number in would require a new engine-read key + run-path wiring. The wizard sets only the existing reproducibility *booleans*.
- Species-subset selection (destructive: cascades to maps, diet, per-species params).
- A new directory-backed preset store (rejected — `ScenarioManager` already persists full config dicts).
- Build-from-scratch ecosystem authoring.

## Architecture

Mirrors the repo's established pattern (Map Builder, FishBase bootstrap): a **browser-free, fully unit-tested pure core** + a **thin Shiny surface** that only does the modal, validation display, and state wiring.

### New pure core: `osmose/scenario_wizard.py`

```python
@dataclass(frozen=True)
class Basics:
    nyear: int
    ndtperyear: int
    reproducible_rng: bool

@dataclass
class BuiltScenario:
    config: dict[str, str]
    config_dir: Path | None
    case_map: dict[str, str]
    parent: str | None   # source scenario name when the source was a saved scenario
```

- `apply_basics(config: dict[str, str], basics: Basics) -> dict[str, str]`
  Returns a NEW dict (shallow copy) with exactly these keys set, everything else untouched:
  - `simulation.time.nyear` = `str(basics.nyear)`
  - `simulation.time.ndtperyear` = `str(basics.ndtperyear)`
  - `movement.randomseed.fixed` = `"true"/"false"` (per `basics.reproducible_rng`)
  - `stochastic.mortality.randomseed.fixed` = `"true"/"false"`

- `read_basics(config: dict[str, str]) -> Basics`
  Pre-fill helper: parse the four keys from a source config (with sane fallbacks — `nyear`/`ndtperyear` default to a positive value if absent/unparseable; `reproducible_rng` = both booleans true).

- `resolve_source(kind: str, name: str, *, scenarios_dir: Path, demo_tmp: Path) -> tuple[dict, Path | None, dict]`
  - `kind == "demo"`: `osmose_demo(name, demo_tmp)` → read the produced master config with `OsmoseConfigReader` → `(config, config_dir=<demo config dir>, case_map=reader.key_case_map)`. Maps/grid resolve because the demo tree is on disk.
  - `kind == "scenario"`: `ScenarioManager(scenarios_dir).load(name)` → `(scenario.config, config_dir=None, scenario.key_case_map)`. `config_dir` is None — scenarios persist only the config dict (no map files), matching today's scenario-load behavior. Returns the source name as `parent`.

- `validate_name(name: str, existing: set[str]) -> list[str]`
  Problems list (empty ⇒ valid): empty/whitespace; contains `/`, `\`, or `..`; already in `existing`. (The `Scenario` dataclass also guards `/`,`\`,`..` in `__post_init__`; this pre-validates for a friendly inline message and adds the duplicate check.)

- `build_scenario(kind, source_name, basics, *, scenarios_dir, demo_tmp) -> BuiltScenario`
  Ties it together: `resolve_source(kind, source_name, ...)` → `apply_basics` → assemble `BuiltScenario(config, config_dir, case_map, parent)`. Does NOT save or touch UI state (the page does that); the new scenario name is supplied separately by the page at save time.

### Page changes: `ui/pages/scenarios.py`

- **UI:** add a prominent `input_action_button("btn_new_scenario", "+ New Scenario", class_="btn-success")` (top of the page / in the Save card area).
- **Modal stepper:** a `wizard_step = reactive.Value(1)` and a `@render.ui` modal body that shows step 1/2/3 controls with Back/Next/Create footer buttons. Step 1 select choices = `{"-- bundled --": {demo: demo}, "-- saved --": {s: s}}` grouped (or a flat select with section labels). On entering step 2, pre-fill the Basics inputs from `read_basics(resolve_source(...).config)` (resolve the source once, cache in a `reactive.Value`).
- **Create handler** (`@reactive.event(input.btn_wizard_create)`):
  1. `errs = validate_name(new_name, {s["name"] for s in mgr.list_scenarios()})`; if errs → inline error in the modal, return.
  2. `built = build_scenario(kind, source_name, basics, scenarios_dir=state.scenarios_dir, demo_tmp=<tempdir>)`.
  3. `mgr.save(Scenario(name=new_name, description=..., config=built.config, key_case_map=built.case_map, parent_scenario=built.parent))` — auto-save.
  4. `state.load_config(built.config, built.case_map)`; if `built.config_dir`: `state.config_dir.set(built.config_dir)`; `state.config_name.set(new_name)`; sync species names (existing pattern); bump `state.load_trigger`; `state.dirty.set(False)`.
  5. Close modal; refresh the scenario list selects; toast a summary.
- Reuse `mgr = ScenarioManager(state.scenarios_dir)` already created in `scenarios_server`.

## Data flow

```
[Scenarios page] --click--> [+ New Scenario]
   -> modal step 1 (Preset)  -> resolve_source() cached
   -> modal step 2 (Basics)  -> read_basics() prefill
   -> modal step 3 (Name)    -> validate_name()
   -> Create:
        build_scenario() -> BuiltScenario
        ScenarioManager.save(Scenario(...))          # persist
        state.load_config(); config_dir; config_name # load into editor
        load_trigger++                                # all pages re-render
   -> modal closes; user is editing the new scenario
```

## Maps caveat (intentional, not a regression)

- **Bundled-demo source:** the demo's full file tree is materialized in a temp dir, `config_dir` points at it, and `movement.file.map{N}` / `grid.mask.file` resolve normally.
- **Saved-scenario source:** scenarios persist only the config dict (no map files), so `config_dir` is `None` and map paths are whatever the scenario's config stored — identical to today's "Scenarios → Load" behavior. Documented, not fixed here.

## Error handling

- Name empty / unsafe (`/`,`\`,`..`) / duplicate → inline modal error, Create blocked.
- `nyear` / `ndtperyear` must parse as integers ≥ 1 → inline error, Create blocked.
- No source selected → Next from step 1 blocked.
- Unknown `kind` → `resolve_source` raises `ValueError` (guarded; surfaced as a toast).
- Demo generation / config read failure → caught, error toast, modal stays open.

## Testing

**Pure core (`tests/test_scenario_wizard.py`):**
- `apply_basics` sets exactly the four keys to the expected string values and leaves an unrelated key untouched; returns a new dict (input unmutated).
- `read_basics` round-trips with `apply_basics`; falls back sanely on missing/garbage values.
- `validate_name`: empty, `../x`, `a/b`, duplicate → each flagged; a clean unique name → `[]`.
- `resolve_source(kind="demo", "baltic", demo_tmp=tmp_path)` → config has `grid.nlon` etc., `config_dir` exists, case_map non-empty.
- `resolve_source(kind="scenario", ...)` against a `ScenarioManager` seeded in `tmp_path` → returns the stored config, `config_dir is None`, `parent == source`.
- `build_scenario` end-to-end (demo source) → config reflects the basics overrides AND retains demo keys; `parent is None`. (scenario source) → `parent == source`.

**Page (`tests/test_ui_scenarios_wizard.py`):**
- `import ui.pages.scenarios` clean; the page exposes the new button id and any extracted helper.
- A small helper test for the source-choices builder (`_wizard_source_choices(demos, scenarios)`).

**e2e (`tests/test_e2e_scenario_wizard.py`, viztest-gated):**
- Load app → Scenarios page → **+ New Scenario** → step 1 pick `baltic` → Next → step 2 set Years → Next → step 3 type a name → **Create** → assert a success toast and the name appears in the scenario list select. (Dismiss the changelog modal first.)

## Files touched

- **Create:** `osmose/scenario_wizard.py`, `tests/test_scenario_wizard.py`, `tests/test_ui_scenarios_wizard.py`, `tests/test_e2e_scenario_wizard.py`.
- **Modify:** `ui/pages/scenarios.py` (button + modal stepper + create handler).
- **No `app.py` change** (Scenarios page already registered) → no nav / visual-baseline change.

## Reused infrastructure

`osmose.demo.osmose_demo` / `list_demos`, `osmose.config.reader.OsmoseConfigReader`, `osmose.scenarios.ScenarioManager` (`save` / `load` / `list_scenarios`) + `Scenario` dataclass, `ui.state` (`load_config`, `config`, `config_dir`, `config_name`, `key_case_map`, `load_trigger`, `dirty`, `scenarios_dir`), and the existing overwrite-confirm modal pattern in `scenarios.py`.
