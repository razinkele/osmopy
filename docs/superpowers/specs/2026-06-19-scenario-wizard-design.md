# Scenario Wizard — Design

**Date:** 2026-06-19
**Status:** Approved (brainstorming complete; revised after in-loop multi-angle review)

## Goal

A guided **"New Scenario"** wizard that lets a user bootstrap a working scenario from a known-good starting point — a bundled demo (Baltic / Bay of Biscay / EEC / EEC Full / minimal) **or** a previously-saved scenario — set a few headline parameters up front, name it, and land in the normal editor with the new scenario already **persisted** and loaded. It closes the gap between today's "Load example" (loads a throwaway temp config in place, no clone/name/save path) and "Scenarios → Fork" (instant auto-named duplicate, no guided edit), giving newcomers a smooth clone-and-customize flow.

## User flow

A **"+ New Scenario"** button on the existing **Scenarios** page opens a modal 3-step stepper:

1. **Preset** — choose a starting point: a bundled demo (from `list_demos()`) or a saved scenario (from `ScenarioManager.list_scenarios()`), shown as two `<optgroup>` sections in one select. A static caption under the select sets expectations: *"Saved scenarios don't include map files, so Grid/Map pages may be empty — bundled demos include full maps."* (A static caption rather than a selection-conditional note, so changing the select never re-renders/resets it.)
2. **Basics** — set headline parameters, each pre-filled from the chosen source's current value:
   - **Years** → `simulation.time.nyear`
   - **Steps/year** → `simulation.time.ndtperyear`
   - **Reproducible runs (Python engine)** (toggle) → sets both `movement.randomseed.fixed` and `stochastic.mortality.randomseed.fixed`
3. **Name** — the new scenario name (validated: non-empty, filesystem-safe, not a duplicate).

**Create** → applies the basics overrides to the (already-resolved, cached) source config → **saves** it as a scenario via `ScenarioManager.save` (auto-save, so it appears in the list immediately) → loads it into the editor (`state.load_config` + `config_dir`/`config_name` + `load_trigger`) → closes the modal. The user is now in the editor on a named, persisted scenario, free to customize further and re-save.

## Relationship to existing Scenarios actions

The page already has Save, Load, **Fork**, Compare, Delete. To keep the mental model clean:

- **Fork** = one-click, no-edit, auto-named (`<name>_fork`) duplicate of a *saved* scenario. Its button is **relabelled "Quick Duplicate"** (label-only change) so it no longer competes with "New Scenario" for the "clone" concept.
- **New Scenario (wizard)** = guided, named, demo-*or*-saved source, with headline overrides. It records lineage the same way Fork does (`parent_scenario` set when the source is a saved scenario).
- **Save** = persist the *current* editor config under a name (overwrite-confirmed on duplicate — unchanged).

This is the only change to existing controls; Fork's behavior is untouched.

## Scope

**In scope:**
- A pure, tested core that resolves a source to a config (+ config_dir + case_map + parent), applies the three headline overrides, and validates a name.
- A modal stepper added to the existing Scenarios page (no new nav tab → no `#main_nav` visual-baseline change).
- Relabel the Fork button to "Quick Duplicate".
- A one-line pointer next to the Grid page's "Load example": *"Want to name and save it? Use Scenarios → + New Scenario."*
- Auto-save-and-load on Create, with an auto-generated default description.

**Out of scope (YAGNI):**
- Seed-*value* input. The engine seed is pinned to `0` at the run path (`ui/pages/run.py:294` passes `seed=0`), so the two `*.randomseed.fixed` booleans alone fully determine repeatability for the Python engine — a user-settable seed is deferred. (Reproducibility is Python-engine only: PCG64 ≠ Java MT19937, per `osmose/engine/rng.py` docstring; hence the toggle label is scoped "(Python engine)".)
- Species-subset selection (destructive: cascades to maps, diet, per-species params).
- A new directory-backed preset store (rejected — `ScenarioManager` already persists full config dicts).
- Build-from-scratch ecosystem authoring.
- Persisting map files inside saved scenarios (the maps caveat below is documented, not fixed).

## Architecture

Mirrors the repo's established pattern (Map Builder, FishBase bootstrap): a **browser-free, fully unit-tested pure core** + a **thin Shiny surface** that does only the modal, validation display, and state wiring. Source resolution happens **once** (on advancing from step 1) and is cached; Create reuses the cached resolution — so a demo tree is generated exactly once and its directory outlives the handler for run-time map resolution.

### New pure core: `osmose/scenario_wizard.py`

```python
@dataclass(frozen=True)
class Basics:
    nyear: int
    ndtperyear: int
    reproducible_rng: bool

@dataclass
class ResolvedSource:
    kind: str                 # "demo" | "scenario" (carried so Create needn't re-parse)
    name: str                 # the source name (demo name or scenario name)
    config: dict[str, str]
    config_dir: Path | None   # demo: the materialized config dir; scenario: None
    case_map: dict[str, str]
    parent: str | None        # source scenario name when kind == "scenario", else None
```

- `parse_source(value: str) -> tuple[str, str]`
  Splits a select value of the form `"demo:<name>"` or `"scenario:<name>"` into `(kind, name)`. Raises `ValueError` on an unknown prefix.

- `source_choices(demos: list[str], scenarios: list[str]) -> dict`
  Returns a Shiny grouped-`<optgroup>` choices dict with prefixed values, e.g.
  `{"Bundled demos": {"demo:baltic": "baltic", ...}, "Saved scenarios": {"scenario:my_run": "my_run", ...}}`.
  Omits the "Saved scenarios" group entirely when `scenarios` is empty (no empty optgroup).

- `resolve_source(kind: str, name: str, *, scenarios_dir: Path, dest_dir: Path | None = None) -> ResolvedSource`
  All branches set `kind` and `name` on the returned `ResolvedSource`.
  - `kind == "demo"`: `osmose_demo(name, dest_dir)` (caller supplies a persistent `dest_dir`) → read the produced master config: `cfg = OsmoseConfigReader().read(result["config_file"])`; `config_dir = result["config_file"].parent`; `case_map = reader.key_case_map`; `parent = None`. (The data path is: demo dict → `config_file` → reader → `(cfg, config_file.parent, key_case_map)`.) Maps/grid resolve because the demo tree is on disk in `dest_dir`.
  - `kind == "scenario"`: `s = ScenarioManager(scenarios_dir).load(name)` → `ResolvedSource(kind, name, s.config, config_dir=None, case_map=s.key_case_map, parent=name)`. `config_dir` is None — scenarios persist only the config dict (no map files), matching today's scenario-load behavior.

- `read_basics(config: dict[str, str]) -> Basics`
  Pre-fill helper. Parses `simulation.time.nyear` / `simulation.time.ndtperyear` (fallbacks: `nyear=10`, `ndtperyear=24` if absent/unparseable) and `reproducible_rng = (movement.randomseed.fixed AND stochastic.mortality.randomseed.fixed both truthy)`.

- `apply_basics(config: dict[str, str], basics: Basics) -> dict[str, str]`
  Returns a NEW dict (shallow copy) with exactly these keys set, everything else untouched:
  - `simulation.time.nyear` = `str(basics.nyear)`
  - `simulation.time.ndtperyear` = `str(basics.ndtperyear)`
  - `movement.randomseed.fixed` = `"true"/"false"`
  - `stochastic.mortality.randomseed.fixed` = `"true"/"false"`

- `validate_name(name: str, existing: set[str]) -> list[str]`
  Problems list (empty ⇒ valid): empty/whitespace; contains `/`, `\`, or `..`; already in `existing`. (The `Scenario` dataclass `__post_init__` also guards `/`,`\`,`..`; this pre-validates for a friendly inline message and adds the duplicate check.)

- `default_description(kind: str, name: str, basics: Basics) -> str`
  e.g. `"Created from baltic demo, 50 yr"` / `"Created from scenario 'my_run', 50 yr"`.

### Page changes: `ui/pages/scenarios.py`

**UI:** add `input_action_button("btn_new_scenario", "+ New Scenario", class_="btn-success")` at the top of the page. Relabel the existing Fork button to "Quick Duplicate" (id unchanged).

**Wizard state (reactive values created in `scenarios_server`):**
- `wizard_step = reactive.Value(1)`
- `wizard_source = reactive.Value[ResolvedSource | None](None)` — the cached resolution + an associated `wizard_source_key = reactive.Value("")` holding the `"kind:name"` it was resolved from
- (the demo temp dir is a local `dest = Path(tempfile.mkdtemp(...))` registered for `atexit` cleanup, not a reactive value)
- `wizard_error = reactive.Value("")`

**Modal mechanism (pinned down; see the implementation plan for the authoritative code):** a single `ui.modal_show(...)`. The body is `@render.ui wizard_body` reading `wizard_step()` (+ `wizard_source()` for step-2 prefill), rendering ONLY the current step's INPUTS (step 1 select + static saved-source caption; step 2 the three Basics inputs prefilled at creation via `value=`; step 3 name input). A SEPARATE `@render.ui wizard_error_msg` reads `wizard_error()` so an error never re-renders/clears the body inputs. The nav buttons live in the modal's STATIC `footer=` (created once per `modal_show`, the proven `btn_confirm_overwrite` pattern) — NOT inside `@render.ui`, since action buttons recreated by a re-rendering `@render.ui` can drop/double-fire clicks. There is a single `btn_wizard_next` (relabelled "Create" on step 3 via `ui.update_action_button`) plus `btn_wizard_back` and a `data-bs-dismiss` Cancel. The modal is removed only on Create-success or Cancel.

**Open ("+ New Scenario") handler** `@reactive.event(input.btn_new_scenario)`: reset `wizard_step=1`, `wizard_source=None`, `wizard_source_key=""`, `wizard_error=""`, clear the name input, then `ui.modal_show(...)`. (Guarantees a clean state on every reopen.)

**Next-from-step-1 handler:** require a selected source else set `wizard_error` and stay. `(kind, name) = parse_source(value)`. If `"kind:name"` differs from `wizard_source_key`: create a fresh persistent dir `dest = Path(tempfile.mkdtemp(prefix="osmose_wizard_"))` (same `mkdtemp` call as `handle_load_example` at `grid.py:819`, but DELIBERATELY without that handler's `atexit.register(shutil.rmtree, ...)` cleanup — `config_dir` must outlive the handler so map paths resolve at run time), store it in `wizard_dest`, `resolved = resolve_source(kind, name, scenarios_dir=state.scenarios_dir, dest_dir=dest)`, cache it in `wizard_source` + set `wizard_source_key`. (Re-selecting the same source reuses the cache; selecting a different one re-resolves.) Then advance `wizard_step=2` — the step-2 `@render.ui` reads the cached `wizard_source` and prefills each input at creation via `value=read_basics(resolved.config).<field>` (no post-creation `update_*` call, so no race). Wrap resolution in try/except → on failure set `wizard_error`, stay on step 1.

**Create** — a `_do_wizard_create()` function called from `_wizard_next`'s step-3 branch (the relabelled "Create" button), NOT a separate `btn_wizard_create`. Its load section is wrapped in `state.busy.set(...)` / `finally` for parity with `handle_load`:
  1. Assemble `basics = Basics(nyear=int(input.wizard_nyear()), ndtperyear=int(input.wizard_ndt()), reproducible_rng=bool(input.wizard_rng()))`, re-checking each parses as an integer ≥1; else `wizard_error`, return.
  2. `new_name = input.wizard_name().strip()`; `errs = validate_name(new_name, {s["name"] for s in mgr.list_scenarios()})`; if errs → `wizard_error = errs[0] + " — try a different name."`, return.
  3. `resolved = wizard_source.get()` (cached); `cfg = apply_basics(resolved.config, basics)`.
  4. `mgr.save(Scenario(name=new_name, description=default_description(resolved.kind, resolved.name, basics), config=cfg, key_case_map=resolved.case_map, parent_scenario=resolved.parent))` — auto-save (`resolved.kind`/`resolved.name` come off the cached `ResolvedSource`, so Create needn't re-parse the select). (Wrapped in try/except → `wizard_error` on failure, modal stays open, no partial state applied.)
  5. Apply to the editor: `state.load_config(cfg, resolved.case_map)` (call for its side effect — its return value is the deprecated-keys list, NOT species names, so ignore it). Then re-derive species names from the config exactly as `scenarios.py:186-194` does (NOT from `load_config`'s return): `n = int(float(cfg.get("simulation.nspecies", "3") or "3"))`; `names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n)]`; `state.species_names.set(names)`; `ui.update_numeric("n_species", value=n)`. If `resolved.config_dir`: `state.config_dir.set(resolved.config_dir)` else leave unchanged; `state.config_name.set(new_name)`; `state.dirty.set(False)`; `state.load_trigger.set(state.load_trigger.get() + 1)`.
  6. `_bump()` (the existing list-refresh trigger at `scenarios.py:77`) so the new scenario appears in the Load/Compare selects; `ui.modal_remove()`; success toast.

Reuse `mgr = ScenarioManager(state.scenarios_dir)` already created in `scenarios_server` (`scenarios.py:74`; `state.scenarios_dir` is a plain `Path`, no `.get()`).

## Data flow

```
[+ New Scenario] -> reset wizard state -> modal_show
  step 1 (Preset)  -> Next: resolve_source() ONCE into persistent dest dir -> cache
                      (saved-source -> inline maps warning)
  step 2 (Basics)  -> prefilled from read_basics(cached.config)
  step 3 (Name)    -> validate_name()
  Create:
    apply_basics(cached.config, basics) -> cfg
    ScenarioManager.save(Scenario(...))            # persist (auto-save)
    state.load_config(cfg); config_dir<-cached;    # load into editor
    config_name; species_names; dirty=False; load_trigger++
    _bump(); modal_remove(); toast
```

## Maps caveat (intentional, not a regression)

- **Bundled-demo source:** the demo's full file tree is materialized in the persistent `dest` dir, `config_dir` points at it, and `movement.file.map{N}` / `grid.mask.file` resolve normally at run time.
- **Saved-scenario source:** scenarios persist only the config dict (no map files), so `config_dir` stays None and map paths are whatever the scenario's config stored — identical to today's "Scenarios → Load". The step-1 inline note sets expectations; not fixed here.

## Error handling

- Name empty / unsafe (`/`,`\`,`..`) / duplicate → inline modal error (`wizard_error`), Create blocked, with a "try a different name" hint.
- `nyear` / `ndtperyear` must parse as integers ≥ 1 → inline error, Create blocked.
- No source selected on step 1 → Next blocked with inline error.
- Unknown source prefix → `parse_source` raises `ValueError` (caught, surfaced inline).
- Demo generation / config read / save failure → caught, inline error, modal stays open, no partial state applied (save precedes any `state` mutation).

## Testing

**Pure core (`tests/test_scenario_wizard.py`):**
- `apply_basics` sets exactly the four keys to the expected string values, leaves an unrelated key untouched, returns a new dict (input unmutated).
- `read_basics` round-trips with `apply_basics`; falls back to `nyear=10`/`ndtperyear=24` on missing/garbage values; `reproducible_rng` true only when both booleans truthy.
- `parse_source` splits `demo:`/`scenario:` correctly; raises on bad prefix.
- `source_choices` groups bundled + saved with prefixed values; omits the saved group when empty. (Also a smoke test that `ui.input_select` accepts the nested-dict `choices` as `<optgroup>`s — no in-repo precedent for nested-dict choices, though Shiny 1.6.3 supports it; verify at build time.)
- `validate_name`: empty, `../x`, `a/b`, duplicate → each flagged; a clean unique name → `[]`.
- `resolve_source(kind="demo", "baltic", scenarios_dir=tmp, dest_dir=tmp2)` → config has `grid.nlon` etc., `config_dir` exists on disk, `case_map` non-empty, `parent is None`, AND `kind == "demo"` / `name == "baltic"` are set on the returned `ResolvedSource`.
- `resolve_source(kind="scenario", ...)` against a `ScenarioManager` seeded in `tmp_path` → returns the stored config, `config_dir is None`, `parent == source`, `case_map` carried, AND `kind == "scenario"` / `name == source` set.
- `default_description` includes the source name and year count.

**Page (`tests/test_ui_scenarios_wizard.py`):**
- `import ui.pages.scenarios` clean; the page exposes `btn_new_scenario` and the extracted helpers.
- `source_choices` integration via the page helper (grouped shape).

**e2e (`tests/test_e2e_scenario_wizard.py`, viztest-gated):**
- Load app → Scenarios page → **+ New Scenario** → step 1 pick `demo:baltic` → Next → step 2 set Years (e.g. 7) → Next → step 3 type a name → **Create** → assert a success toast, the name appears in the Load select, AND (load that scenario / read it back) the saved config's `simulation.time.nyear == "7"` (proves `apply_basics` reached the save path, not just that the name appeared). Dismiss the changelog modal first.

## Files touched

- **Create:** `osmose/scenario_wizard.py`, `tests/test_scenario_wizard.py`, `tests/test_ui_scenarios_wizard.py`, `tests/test_e2e_scenario_wizard.py`.
- **Modify:** `ui/pages/scenarios.py` (button + Fork relabel + modal stepper + handlers), `ui/pages/grid.py` (one-line pointer next to "Load example").
- **No `app.py` change** (Scenarios page already registered) → no nav / visual-baseline change.

## Reused infrastructure

`osmose.demo.osmose_demo` / `list_demos`, `osmose.config.reader.OsmoseConfigReader` (`read` + `key_case_map`), `osmose.scenarios.ScenarioManager` (`save` / `load` / `list_scenarios`) + `Scenario` dataclass (note: `ScenarioManager.load` and `state.load_config` both canonicalize — idempotent, harmless double-canonicalize for scenario sources), `ui.state` (`load_config`, `config`, `config_dir`, `config_name`, `key_case_map`, `species_names`, `load_trigger`, `dirty`, `scenarios_dir`), and the existing `_bump()` list-refresh trigger in `scenarios.py`.
