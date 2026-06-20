# Scenario Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A guided "+ New Scenario" modal wizard on the Scenarios page that bootstraps a working scenario from a bundled demo or a saved scenario, sets headline params (years / steps-per-year / reproducible-RNG), names it, then auto-saves and loads it into the editor.

**Architecture:** A browser-free, fully unit-tested pure core (`osmose/scenario_wizard.py`: source resolution, basics overrides, name validation, choice helpers) + a thin modal stepper added to the existing `ui/pages/scenarios.py`. Source resolution happens once (on advancing from step 1) into a persistent temp dir and is cached; Create reuses the cache. Reuses `osmose_demo`, `OsmoseConfigReader`, `ScenarioManager`, and `state.load_config`.

**Tech Stack:** Python 3.12, Shiny for Python 1.6.3, pytest (+ Playwright for the viztest-gated e2e). Run with `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-19-scenario-wizard-design.md` (approved, 3-round-reviewed).

## File structure

- Create `osmose/scenario_wizard.py` — the pure core (dataclasses + 7 functions).
- Modify `ui/pages/scenarios.py` — add the "+ New Scenario" button, relabel Fork → "Quick Duplicate", add the modal stepper (state + `wizard_body`/`wizard_error_msg` renderers + open/back/next/create handlers).
- Modify `ui/pages/grid.py` — one-line pointer next to "Load example".
- Tests: `tests/test_scenario_wizard.py` (pure core), `tests/test_ui_scenarios_wizard.py` (page smoke), `tests/test_e2e_scenario_wizard.py` (draw-seam e2e, viztest-gated).
- No `app.py` change (Scenarios page already registered) → no nav / visual-baseline change.

## Key verified facts

- Config keys: `simulation.time.nyear`, `simulation.time.ndtperyear`, `movement.randomseed.fixed`, `stochastic.mortality.randomseed.fixed`. RNG seed *value* is a run-time arg (`ui/pages/run.py:294 seed=0`), not a config key — wizard sets only the two booleans.
- `osmose.demo.osmose_demo(scenario, output_dir) -> {"config_file","output_dir"}`; `list_demos() -> ["baltic","bay_of_biscay","eec","eec_full","minimal"]`.
- `OsmoseConfigReader().read(config_file) -> dict`; `.key_case_map` populated after `read`.
- `ScenarioManager(storage_dir)` with `.save(scenario)`, `.load(name) -> Scenario`, `.list_scenarios() -> list[dict]` (each has `["name"]`). `Scenario(name, description, config, key_case_map, parent_scenario, tags)`; `__post_init__` rejects empty / `/` / `\` / `..`.
- `state` (ui/state.py): `load_config(cfg, case_map=None) -> list[str]` (returns DEPRECATED KEYS, not species names), `config`, `config_dir` (reactive Path|None), `config_name`, `key_case_map`, `species_names`, `load_trigger`, `dirty`; `scenarios_dir` is a plain `Path`.
- `ui/pages/scenarios.py`: `mgr = ScenarioManager(state.scenarios_dir)` (`:74`), `_bump()` refresh trigger (`:77`), `_scenario_names()` (`:83`), species-sync pattern (`:186-194`), Fork button (`:37-40`).
- Shiny 1.6.3: `input_select` accepts nested-dict `choices` as `<optgroup>`; `input_switch`/`update_switch`, `input_numeric`/`update_numeric`, `input_text`/`update_text` all exist and are used in-repo.

---

## Task 1: pure core scaffold + `Basics`/`ResolvedSource` + `apply_basics`

**Files:**
- Create: `osmose/scenario_wizard.py`
- Test: `tests/test_scenario_wizard.py`

- [ ] **Step 1: Write the failing test**

```python
from osmose.scenario_wizard import Basics, apply_basics


def test_apply_basics_sets_exactly_the_four_keys_and_copies():
    cfg = {"simulation.time.nyear": "100", "species.name.sp0": "cod"}
    out = apply_basics(cfg, Basics(nyear=50, ndtperyear=12, reproducible_rng=True))
    assert out["simulation.time.nyear"] == "50"
    assert out["simulation.time.ndtperyear"] == "12"
    assert out["movement.randomseed.fixed"] == "true"
    assert out["stochastic.mortality.randomseed.fixed"] == "true"
    assert out["species.name.sp0"] == "cod"  # untouched
    assert cfg["simulation.time.nyear"] == "100"  # input not mutated


def test_apply_basics_false_rng():
    out = apply_basics({}, Basics(nyear=10, ndtperyear=24, reproducible_rng=False))
    assert out["movement.randomseed.fixed"] == "false"
    assert out["stochastic.mortality.randomseed.fixed"] == "false"
```

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_scenario_wizard.py -k apply_basics -q` → ImportError / fail.

- [ ] **Step 3: Implement** `osmose/scenario_wizard.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_NYEAR_KEY = "simulation.time.nyear"
_NDT_KEY = "simulation.time.ndtperyear"
_MOVE_RNG_KEY = "movement.randomseed.fixed"
_MORT_RNG_KEY = "stochastic.mortality.randomseed.fixed"
_DEFAULT_NYEAR = 10
_DEFAULT_NDT = 24


@dataclass(frozen=True)
class Basics:
    nyear: int
    ndtperyear: int
    reproducible_rng: bool


@dataclass
class ResolvedSource:
    kind: str  # "demo" | "scenario"
    name: str
    config: dict[str, str]
    config_dir: Path | None
    case_map: dict[str, str]
    parent: str | None


def apply_basics(config: dict[str, str], basics: Basics) -> dict[str, str]:
    """Return a new config with the four headline keys set; everything else untouched."""
    out = dict(config)
    out[_NYEAR_KEY] = str(basics.nyear)
    out[_NDT_KEY] = str(basics.ndtperyear)
    flag = "true" if basics.reproducible_rng else "false"
    out[_MOVE_RNG_KEY] = flag
    out[_MORT_RNG_KEY] = flag
    return out
```

- [ ] **Step 4: Run, verify PASS.** Same `-k apply_basics`. `.venv/bin/ruff check osmose/scenario_wizard.py tests/test_scenario_wizard.py` + `.venv/bin/ruff format osmose/scenario_wizard.py tests/test_scenario_wizard.py` clean.

- [ ] **Step 5: Commit**

```bash
git add osmose/scenario_wizard.py tests/test_scenario_wizard.py
git commit -m "feat(wizard): Basics/ResolvedSource + apply_basics core

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `read_basics`

**Files:** Modify `osmose/scenario_wizard.py`; Test: `tests/test_scenario_wizard.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.scenario_wizard import read_basics


def test_read_basics_roundtrips_with_apply_basics():
    cfg = apply_basics({}, Basics(nyear=33, ndtperyear=12, reproducible_rng=True))
    assert read_basics(cfg) == Basics(nyear=33, ndtperyear=12, reproducible_rng=True)


def test_read_basics_falls_back_on_missing_or_garbage():
    assert read_basics({}) == Basics(nyear=10, ndtperyear=24, reproducible_rng=False)
    assert read_basics({"simulation.time.nyear": "x"}).nyear == 10


def test_read_basics_rng_true_only_when_both_booleans_true():
    assert read_basics({"movement.randomseed.fixed": "true"}).reproducible_rng is False
    both = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
    assert read_basics(both).reproducible_rng is True
```

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_scenario_wizard.py -k read_basics -q`

- [ ] **Step 3: Implement** (append):

```python
def _to_int(value: object, default: int) -> int:
    try:
        n = int(float(str(value)))
    except (ValueError, TypeError):
        return default
    return n if n >= 1 else default


def read_basics(config: dict[str, str]) -> Basics:
    """Parse the four headline keys from a config (sane fallbacks for missing/garbage)."""
    move = str(config.get(_MOVE_RNG_KEY, "false")).lower() == "true"
    mort = str(config.get(_MORT_RNG_KEY, "false")).lower() == "true"
    return Basics(
        nyear=_to_int(config.get(_NYEAR_KEY), _DEFAULT_NYEAR),
        ndtperyear=_to_int(config.get(_NDT_KEY), _DEFAULT_NDT),
        reproducible_rng=move and mort,
    )
```

- [ ] **Step 4: Run, verify PASS.** Same `-k read_basics`; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add osmose/scenario_wizard.py tests/test_scenario_wizard.py
git commit -m "feat(wizard): read_basics prefill helper"
```

---

## Task 3: `parse_source` + `source_choices`

**Files:** Modify `osmose/scenario_wizard.py`; Test: `tests/test_scenario_wizard.py`

- [ ] **Step 1: Write the failing test** (append)

```python
import pytest

from osmose.scenario_wizard import parse_source, source_choices


def test_parse_source():
    assert parse_source("demo:baltic") == ("demo", "baltic")
    assert parse_source("scenario:my_run") == ("scenario", "my_run")
    with pytest.raises(ValueError):
        parse_source("bogus")


def test_source_choices_groups_and_prefixes():
    ch = source_choices(["baltic", "eec"], ["my_run"])
    assert ch["Bundled demos"] == {"demo:baltic": "baltic", "demo:eec": "eec"}
    assert ch["Saved scenarios"] == {"scenario:my_run": "my_run"}


def test_source_choices_omits_saved_group_when_empty():
    ch = source_choices(["baltic"], [])
    assert "Saved scenarios" not in ch
    assert ch["Bundled demos"] == {"demo:baltic": "baltic"}
```

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_scenario_wizard.py -k "parse_source or source_choices" -q`

- [ ] **Step 3: Implement** (append):

```python
def parse_source(value: str) -> tuple[str, str]:
    """Split a select value 'demo:<name>' / 'scenario:<name>' into (kind, name)."""
    for kind in ("demo", "scenario"):
        prefix = f"{kind}:"
        if value.startswith(prefix):
            return (kind, value[len(prefix) :])
    raise ValueError(f"unknown source value: {value!r}")


def source_choices(demos: list[str], scenarios: list[str]) -> dict[str, dict[str, str]]:
    """Grouped <optgroup> choices for input_select; omit the saved group when empty."""
    choices: dict[str, dict[str, str]] = {"Bundled demos": {f"demo:{d}": d for d in demos}}
    if scenarios:
        choices["Saved scenarios"] = {f"scenario:{s}": s for s in scenarios}
    return choices
```

- [ ] **Step 4: Run, verify PASS.** Same `-k`; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add osmose/scenario_wizard.py tests/test_scenario_wizard.py
git commit -m "feat(wizard): parse_source + grouped source_choices"
```

---

## Task 4: `validate_name` + `default_description`

**Files:** Modify `osmose/scenario_wizard.py`; Test: `tests/test_scenario_wizard.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.scenario_wizard import default_description, validate_name


def test_validate_name():
    existing = {"baltic_run"}
    assert validate_name("new_run", existing) == []
    assert validate_name("", existing)
    assert validate_name("   ", existing)
    assert validate_name("../evil", existing)
    assert validate_name("a/b", existing)
    assert validate_name("a\\b", existing)
    assert validate_name("baltic_run", existing)  # duplicate


def test_default_description():
    b = Basics(nyear=50, ndtperyear=24, reproducible_rng=False)
    assert default_description("demo", "baltic", b) == "Created from baltic demo, 50 yr"
    assert default_description("scenario", "my_run", b) == "Created from scenario 'my_run', 50 yr"
```

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_scenario_wizard.py -k "validate_name or default_description" -q`

- [ ] **Step 3: Implement** (append):

```python
def validate_name(name: str, existing: set[str]) -> list[str]:
    """Problems with a proposed scenario name (empty list = valid)."""
    problems: list[str] = []
    n = (name or "").strip()
    if not n:
        return ["Name must not be empty"]
    if "/" in n or "\\" in n or ".." in n:
        problems.append(f"Name contains invalid characters: {n!r}")
    if n in existing:
        problems.append(f"A scenario named '{n}' already exists")
    return problems


def default_description(kind: str, name: str, basics: Basics) -> str:
    src = f"{name} demo" if kind == "demo" else f"scenario '{name}'"
    return f"Created from {src}, {basics.nyear} yr"
```

- [ ] **Step 4: Run, verify PASS.** Same `-k`; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add osmose/scenario_wizard.py tests/test_scenario_wizard.py
git commit -m "feat(wizard): validate_name + default_description"
```

---

## Task 5: `resolve_source` (demo + scenario branches)

**Files:** Modify `osmose/scenario_wizard.py`; Test: `tests/test_scenario_wizard.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from osmose.scenario_wizard import resolve_source


def test_resolve_source_demo(tmp_path):
    dest = tmp_path / "demo_dest"
    dest.mkdir()
    r = resolve_source("demo", "baltic", scenarios_dir=tmp_path / "scen", dest_dir=dest)
    assert r.kind == "demo" and r.name == "baltic"
    assert r.parent is None
    assert r.config_dir is not None and r.config_dir.exists()
    assert "grid.nlon" in r.config and r.case_map  # real demo config + case map


def test_resolve_source_scenario(tmp_path):
    from osmose.scenarios import Scenario, ScenarioManager

    scen_dir = tmp_path / "scen"
    mgr = ScenarioManager(scen_dir)
    mgr.save(Scenario(name="base", config={"simulation.nspecies": "2"}, key_case_map={"a": "A"}))
    r = resolve_source("scenario", "base", scenarios_dir=scen_dir, dest_dir=None)
    assert r.kind == "scenario" and r.name == "base"
    assert r.config_dir is None
    assert r.parent == "base"
    assert r.config["simulation.nspecies"] == "2"


def test_resolve_source_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        resolve_source("bogus", "x", scenarios_dir=tmp_path, dest_dir=None)
```

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_scenario_wizard.py -k resolve_source -q`

- [ ] **Step 3: Implement** (append). Imports go at the TOP of the file with the others (`from osmose.demo import osmose_demo`, `from osmose.config.reader import OsmoseConfigReader`, `from osmose.scenarios import ScenarioManager`):

```python
def resolve_source(
    kind: str,
    name: str,
    *,
    scenarios_dir: Path,
    dest_dir: Path | None = None,
) -> ResolvedSource:
    """Resolve a wizard source to a config (+ dir + case_map + parent).

    demo: materialize into `dest_dir` (caller-owned, persistent) and read it.
    scenario: load the stored config dict (no files; config_dir is None).
    """
    if kind == "demo":
        if dest_dir is None:
            raise ValueError("dest_dir is required for a demo source")
        result = osmose_demo(name, dest_dir)
        config_file = Path(result["config_file"])
        reader = OsmoseConfigReader()
        cfg = reader.read(config_file)
        return ResolvedSource(
            kind="demo",
            name=name,
            config=cfg,
            config_dir=config_file.parent,
            case_map=dict(reader.key_case_map),
            parent=None,
        )
    if kind == "scenario":
        s = ScenarioManager(scenarios_dir).load(name)
        return ResolvedSource(
            kind="scenario",
            name=name,
            config=dict(s.config),
            config_dir=None,
            case_map=dict(s.key_case_map),
            parent=name,
        )
    raise ValueError(f"unknown source kind: {kind!r}")
```

- [ ] **Step 4: Run, verify PASS.** `.venv/bin/python -m pytest tests/test_scenario_wizard.py -q` (whole file green). `.venv/bin/ruff check osmose/scenario_wizard.py tests/test_scenario_wizard.py` + `.venv/bin/ruff format --check osmose/scenario_wizard.py tests/test_scenario_wizard.py` clean. `.venv/bin/pyright --pythonpath .venv/bin/python osmose/scenario_wizard.py` → 0 errors.

- [ ] **Step 5: Commit**

```bash
git add osmose/scenario_wizard.py tests/test_scenario_wizard.py
git commit -m "feat(wizard): resolve_source (demo materialize + scenario load)"
```

---

## Task 6: page — "+ New Scenario" button + Fork relabel + page smoke test

**Files:**
- Modify: `ui/pages/scenarios.py`
- Test: `tests/test_ui_scenarios_wizard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_scenarios_page_has_new_scenario_button_and_quick_duplicate():
    import ui.pages.scenarios as sc

    assert hasattr(sc, "scenarios_ui") and hasattr(sc, "scenarios_server")
    html = str(sc.scenarios_ui())
    assert "btn_new_scenario" in html
    assert "New Scenario" in html
    assert "Quick Duplicate" in html  # Fork relabelled
    assert ">Fork<" not in html  # old label gone
```

- [ ] **Step 2: Run, verify FAIL.** `.venv/bin/python -m pytest tests/test_ui_scenarios_wizard.py -q`

- [ ] **Step 3: Implement** in `ui/pages/scenarios.py`. (a) In `scenarios_ui()`, add the New-Scenario button immediately after `expand_tab("Save Scenario", "scenarios"),` (line 18):

```python
        expand_tab("Save Scenario", "scenarios"),
        ui.input_action_button(
            "btn_new_scenario", "+ New Scenario", class_="btn-success mb-3"
        ),
```

(b) Relabel the Fork button — change ONLY the label literal `"Fork"` → `"Quick Duplicate"`, keep the id `btn_fork_scenario`. The real source is a single line at `ui/pages/scenarios.py:37`; Edit that exact line:

```python
# old (scenarios.py:37 — one line):
                    ui.input_action_button("btn_fork_scenario", "Fork", class_="btn-info w-100"),
# new (after the label change this line is 108 cols, so `ruff format` in Step 4 will
# auto-wrap it to a 3-line call — that's expected, let the formatter do it):
                    ui.input_action_button("btn_fork_scenario", "Quick Duplicate", class_="btn-info w-100"),
```

- [ ] **Step 4: Run, verify PASS.** `.venv/bin/python -m pytest tests/test_ui_scenarios_wizard.py -q`; `.venv/bin/python -c "import app"` clean; ruff check/format clean on `ui/pages/scenarios.py tests/test_ui_scenarios_wizard.py`.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/scenarios.py tests/test_ui_scenarios_wizard.py
git commit -m "feat(ui): scenarios page New-Scenario button + Fork relabel"
```

---

## Task 7: page — wizard state + modal + open/back/next handlers

**Files:** Modify `ui/pages/scenarios.py` (test: import-clean gate — the reactive/modal seam is exercised by the Task 10 e2e)

**Design note (why a static footer):** the Back/Next nav buttons live in the modal's STATIC `footer=` (created once per `modal_show`, exactly like the existing `btn_confirm_overwrite` overwrite-modal at `scenarios.py:125-141`), NOT inside the `@render.ui wizard_body`. Action buttons recreated by a re-rendering `@render.ui` can have their click-counter rebound (swallowed clicks / spurious double-advance) — so `wizard_body` renders ONLY the per-step inputs, and the nav buttons are stable across step changes. A single "Next" button advances/creates depending on `wizard_step` (relabelled "Create" on step 3).

- [ ] **Step 1: Implement.** Add imports at the TOP of `ui/pages/scenarios.py` (with the existing imports — note `Scenario`, `reactive`, `render`, `ui`, and the module-level `_log` are ALREADY imported at `scenarios.py:6-13`):

```python
import atexit
import shutil
import tempfile

from osmose.demo import list_demos
from osmose.scenario_wizard import (
    Basics,
    ResolvedSource,
    apply_basics,
    default_description,
    parse_source,
    read_basics,
    resolve_source,
    source_choices,
    validate_name,
)
```

Then, inside `scenarios_server(...)` (after `_scenario_names` is defined, ~line 86), add the wizard state, the two renderers, and the open/back/next handlers. The nav buttons are in the modal footer (static); `wizard_body` renders inputs only:

```python
    # --- New Scenario wizard ---
    wizard_step = reactive.Value(1)
    wizard_source: reactive.Value[ResolvedSource | None] = reactive.Value(None)
    wizard_source_key = reactive.Value("")
    wizard_error = reactive.Value("")

    @render.ui
    def wizard_error_msg():
        msg = wizard_error.get()
        return ui.div(msg, class_="text-danger mb-2") if msg else None

    @render.ui
    def wizard_body():
        step = wizard_step.get()
        if step == 1:
            choices = source_choices(list_demos(), _scenario_names())
            return ui.div(
                ui.input_select("wizard_source_sel", "Start from", choices=choices),
                ui.div(
                    "Saved scenarios don't include map files, so Grid/Map pages may be "
                    "empty — bundled demos include full maps.",
                    class_="text-muted small mb-2",
                ),
            )
        if step == 2:
            src = wizard_source.get()
            b = read_basics(src.config) if src is not None else read_basics({})
            return ui.div(
                ui.input_numeric("wizard_nyear", "Years", value=b.nyear, min=1),
                ui.input_numeric("wizard_ndt", "Steps/year", value=b.ndtperyear, min=1),
                ui.input_switch(
                    "wizard_rng", "Reproducible runs (Python engine)",
                    value=b.reproducible_rng,
                ),
            )
        return ui.div(
            ui.input_text("wizard_name", "New scenario name"),
        )

    @reactive.effect
    @reactive.event(input.btn_new_scenario)
    def _wizard_open():
        wizard_step.set(1)
        wizard_source.set(None)
        wizard_source_key.set("")
        wizard_error.set("")
        ui.modal_show(
            ui.modal(
                ui.output_ui("wizard_error_msg"),
                ui.output_ui("wizard_body"),
                title="New Scenario",
                easy_close=False,
                footer=ui.div(
                    ui.tags.button(
                        "Cancel", class_="btn btn-secondary",
                        **{"data-bs-dismiss": "modal"},
                    ),
                    ui.input_action_button(
                        "btn_wizard_back", "Back", class_="btn-secondary"
                    ),
                    ui.input_action_button(
                        "btn_wizard_next", "Next", class_="btn-primary"
                    ),
                    class_="d-flex gap-2",
                ),
            )
        )

    @reactive.effect
    @reactive.event(input.btn_wizard_back)
    def _wizard_back():
        wizard_error.set("")
        step = wizard_step.get()
        if step >= 2:
            if step == 3:
                ui.update_action_button("btn_wizard_next", label="Next")
            wizard_step.set(step - 1)

    @reactive.effect
    @reactive.event(input.btn_wizard_next)
    def _wizard_next():
        wizard_error.set("")
        step = wizard_step.get()
        if step == 1:
            value = input.wizard_source_sel()
            if not value:
                wizard_error.set("Pick a starting point.")
                return
            kind, name = parse_source(value)
            key = f"{kind}:{name}"
            if key != wizard_source_key.get() or wizard_source.get() is None:
                try:
                    dest = None
                    if kind == "demo":
                        dest = Path(tempfile.mkdtemp(prefix="osmose_wizard_"))
                        atexit.register(shutil.rmtree, str(dest), True)
                    resolved = resolve_source(
                        kind, name, scenarios_dir=state.scenarios_dir, dest_dir=dest
                    )
                except (OSError, ValueError, KeyError) as exc:
                    _log.error("wizard resolve failed: %s", exc, exc_info=True)
                    wizard_error.set("Could not load that source. Check server logs.")
                    return
                wizard_source.set(resolved)
                wizard_source_key.set(key)
            wizard_step.set(2)
        elif step == 2:
            wizard_step.set(3)
            ui.update_action_button("btn_wizard_next", label="Create")
        elif step == 3:
            pass  # Create wired in Task 8 (replaced with _do_wizard_create())
```

NOTES:
- The nav buttons (`btn_wizard_back`/`btn_wizard_next`) are in the STATIC modal footer — created once per `modal_show`, never recreated by a step change — so clicks are reliable (the proven `btn_confirm_overwrite` pattern). The single "Next" button is relabelled "Create" on step 3 via `ui.update_action_button` and back to "Next" on Back from step 3.
- `wizard_body` depends on `wizard_step` + `wizard_source` only (NOT `wizard_error`), so a validation error re-renders `wizard_error_msg` alone and never clears the user's typed inputs.
- Step-2 inputs are prefilled directly from the cached source in the `@render.ui` (`value=b.nyear` at creation — no `update_numeric` race); `read_basics({})` supplies the 10/24/False fallback when no source (single source of truth for the defaults).
- The demo tempdir is registered for `atexit` cleanup (matching `export_all_scenarios`), so it's removed at process exit but still outlives the handler for run-time map resolution.
- The step-3 branch is `pass` at this checkpoint (the wizard is fully navigable; Create is wired in Task 8). No undefined-symbol reference, so `import app` + pyright are clean at this commit.

- [ ] **Step 2: Verify.** `.venv/bin/python -c "import app"` clean; `.venv/bin/ruff check ui/pages/scenarios.py` + `.venv/bin/ruff format --check ui/pages/scenarios.py` clean; `.venv/bin/pyright --pythonpath .venv/bin/python ui/pages/scenarios.py` → 0 NEW errors.

- [ ] **Step 3: Commit**

```bash
git add ui/pages/scenarios.py
git commit -m "feat(ui): scenario wizard modal stepper (static footer, state, body, nav)"
```

---

## Task 8: page — Create handler (apply + save + load + close)

**Files:** Modify `ui/pages/scenarios.py` (test: import-clean gate + Task 10 e2e)

- [ ] **Step 1: Implement.** Add the `_do_wizard_create()` function inside `scenarios_server(...)`, after `_wizard_next` (Task 7). It is called from the step-3 branch of `_wizard_next` (see Step 2). The load section is wrapped in `state.busy` for parity with `handle_load` (so `sync_inputs` doesn't fire mid-update):

```python
    def _do_wizard_create():
        wizard_error.set("")
        resolved = wizard_source.get()
        if resolved is None:
            wizard_error.set("No source resolved — go back to step 1.")
            return
        try:
            nyear = int(float(input.wizard_nyear()))
            ndt = int(float(input.wizard_ndt()))
        except (ValueError, TypeError):
            wizard_error.set("Years and Steps/year must be integers.")
            return
        if nyear < 1 or ndt < 1:
            wizard_error.set("Years and Steps/year must be at least 1.")
            return
        basics = Basics(nyear=nyear, ndtperyear=ndt, reproducible_rng=bool(input.wizard_rng()))
        new_name = (input.wizard_name() or "").strip()
        errs = validate_name(new_name, set(_scenario_names()))
        if errs:
            wizard_error.set(errs[0] + " — try a different name.")
            return
        cfg = apply_basics(resolved.config, basics)
        scenario = Scenario(
            name=new_name,
            description=default_description(resolved.kind, resolved.name, basics),
            config=cfg,
            key_case_map=dict(resolved.case_map),
            parent_scenario=resolved.parent,
        )
        try:
            mgr.save(scenario)
        except (OSError, ValueError) as exc:
            _log.error("wizard save failed: %s", exc, exc_info=True)
            wizard_error.set("Could not save the scenario. Check server logs.")
            return
        # Load into the editor. load_config's RETURN is the deprecated-keys list (NOT names);
        # re-derive species names from cfg, exactly as handle_load (scenarios.py:186-194).
        state.busy.set(f"Creating '{new_name}'…")
        try:
            state.load_config(cfg)
            state.config_name.set(new_name)
            state.key_case_map.set(dict(resolved.case_map))
            if resolved.config_dir is not None:
                state.config_dir.set(resolved.config_dir)
            state.dirty.set(False)
            try:
                n_species = int(float(cfg.get("simulation.nspecies", "3") or "3"))
            except (ValueError, TypeError):
                n_species = 3
            names = [cfg.get(f"species.name.sp{i}", f"Species {i}") for i in range(n_species)]
            state.species_names.set(names)
            ui.update_numeric("n_species", value=n_species)
            with reactive.isolate():
                state.load_trigger.set(state.load_trigger.get() + 1)
        finally:
            state.busy.set(None)
        _bump()
        ui.modal_remove()
        ui.notification_show(f"Created scenario '{new_name}'.", type="message", duration=4)
```

- [ ] **Step 2: Wire it into `_wizard_next`.** In the `_wizard_next` handler from Task 7, replace the step-3 placeholder:

```python
        elif step == 3:
            pass  # Create wired in Task 8 (replaced with _do_wizard_create())
```

with:

```python
        elif step == 3:
            _do_wizard_create()
```

- [ ] **Step 3: Verify.** `.venv/bin/python -c "import app"` clean; ruff check/format clean on `ui/pages/scenarios.py`; `.venv/bin/pyright --pythonpath .venv/bin/python ui/pages/scenarios.py` → 0 errors.

- [ ] **Step 4: Commit**

```bash
git add ui/pages/scenarios.py
git commit -m "feat(ui): scenario wizard Create handler (apply + auto-save + load)"
```

---

## Task 9: grid.py — pointer next to "Load example"

**Files:** Modify `ui/pages/grid.py`

- [ ] **Step 1: Implement.** In `ui/pages/grid.py` `grid_ui()` (def at line 75), the "Grid Type" card holds a `ui.div(ui.layout_columns(load_example + btn_load_example ...))` block whose inner div ends ~line 118, immediately followed by `ui.hr()` (line 120). Insert the muted pointer as a new sibling element BETWEEN that `ui.div(...)` block and `ui.hr()` (i.e. just before the `ui.hr()`), matching the surrounding indentation:

```python
                ui.div(
                    "Want to name and save it? Use Scenarios → + New Scenario.",
                    class_="text-muted small mt-1",
                ),
```

- [ ] **Step 2: Verify.** `.venv/bin/python -c "import app"` clean; `.venv/bin/python -c "import ui.pages.grid as g; assert 'New Scenario' in str(g.grid_ui()); print('OK')"` → `OK`. ruff check/format clean on `ui/pages/grid.py`.

- [ ] **Step 3: Commit**

```bash
git add ui/pages/grid.py
git commit -m "feat(ui): point Grid 'Load example' at the Scenarios wizard"
```

---

## Task 10: e2e — wizard flow (viztest-gated)

**Files:** Create `tests/test_e2e_scenario_wizard.py`

- [ ] **Step 1: Write the e2e test** (named `test_e2e_*` so conftest collect-ignore + `[viztest]` gating apply). The nav is a single "Next" button (relabelled "Create" on step 3), so `#btn_wizard_next` is clicked three times. After Create, the test reads back the saved `scenario.json` and asserts `simulation.time.nyear == "7"` (proves `apply_basics` reached the save path — the spec's key assertion), using a unique name + cleanup so re-runs don't collide:

```python
"""End-to-end test for the New Scenario wizard.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_scenario_wizard.py -v -m e2e

Excluded from the default suite (`-m 'not e2e'`). The wizard's pure logic
(apply_basics override, validation, resolve) is covered by
tests/test_scenario_wizard.py; this asserts the modal stepper flow end to end
plus that the override actually reached the persisted config.
"""

import json
import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_SCENARIOS_DIR = Path("data/scenarios")  # state.scenarios_dir default (ui/state.py:35)


def _goto_scenarios(page: Page, app: ShinyAppProc) -> None:
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
    page.wait_for_selector("#btn_new_scenario", timeout=_LOAD_TIMEOUT)


def test_wizard_creates_scenario_from_demo(page: Page, app: ShinyAppProc):
    name = f"e2e_wiz_{uuid.uuid4().hex[:8]}"
    scen_dir = _SCENARIOS_DIR / name
    try:
        _goto_scenarios(page, app)
        # Open the wizard
        page.click("#btn_new_scenario")
        page.wait_for_selector("#wizard_source_sel", timeout=_LOAD_TIMEOUT)
        # Step 1: pick a bundled demo, Next
        page.select_option("#wizard_source_sel", "demo:baltic")
        page.click("#btn_wizard_next")
        # Step 2: set years, Next
        page.wait_for_selector("#wizard_nyear", timeout=_LOAD_TIMEOUT)
        page.fill("#wizard_nyear", "7")
        page.click("#btn_wizard_next")
        # Step 3: name, then Create (the same button, now labelled "Create")
        page.wait_for_selector("#wizard_name", timeout=_LOAD_TIMEOUT)
        page.fill("#wizard_name", name)
        page.click("#btn_wizard_next")
        # Success: toast + the new scenario appears in the list
        note = page.locator(".shiny-notification").last
        expect(note).to_be_visible(timeout=_LOAD_TIMEOUT)
        expect(page.get_by_text(name)).to_be_visible(timeout=_LOAD_TIMEOUT)
        # Read back the persisted config — the Years override must have reached the save path.
        scen_json = scen_dir / "scenario.json"
        assert scen_json.exists(), f"expected saved scenario at {scen_json}"
        saved = json.loads(scen_json.read_text())
        assert saved["config"]["simulation.time.nyear"] == "7"
    finally:
        shutil.rmtree(scen_dir, ignore_errors=True)
```

- [ ] **Step 2: Run (if Playwright/chromium available).** `.venv/bin/python -m pytest tests/test_e2e_scenario_wizard.py -v -m e2e`. Expected: PASS. If the browser is unavailable, document the manual check (open Scenarios → + New Scenario → baltic → Next → years 7 → Next → name → Create → toast + name in list).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_scenario_wizard.py
git commit -m "test(wizard): e2e New-Scenario flow (Baltic demo)"
```

---

## Task 11: final gates

**Files:** none (verification) — fix-and-commit only if a gate fails.

- [ ] **Step 1: Full suite.** `.venv/bin/python -m pytest -q -m "not e2e and not visual" -n auto` → report counts. New `tests/test_scenario_wizard.py` + `tests/test_ui_scenarios_wizard.py` green; nothing else regressed.

- [ ] **Step 2: Lint/format/pyright.** `.venv/bin/ruff check osmose/ ui/ tests/` + `.venv/bin/ruff format --check osmose/ ui/ tests/` clean; `.venv/bin/pyright --pythonpath .venv/bin/python osmose/scenario_wizard.py ui/pages/scenarios.py ui/pages/grid.py` → 0 NEW errors. `.venv/bin/python -c "import app"` clean.

- [ ] **Step 3: e2e (optional).** `.venv/bin/python -m pytest tests/test_e2e_scenario_wizard.py -v -m e2e` if a browser is available; else note the manual check.

- [ ] **Step 4: Commit any gate fixes** (explicit paths; not `git add -A`).

```bash
git add <changed files>
git commit -m "test(wizard): final gates"
```

---

## Notes

- **The pure core (Tasks 1–5) is fully CI-tested**; the page (Tasks 6–8) keeps all logic in the core, so its tests are the page smoke (Task 6) + the e2e (Task 10) for the modal/reactive seam.
- **No new runtime deps** — reuses numpy-free stdlib + existing osmose/ui modules.
- **No `app.py` / nav change** → no `#main_nav` visual-baseline change (unlike the Map Builder).
- **Reproducibility is Python-engine only** (PCG64 ≠ Java MT19937) — the toggle label says "(Python engine)"; the seed value stays pinned to 0 at the run path (out of scope to make settable).
- **Maps caveat:** a saved-scenario source has `config_dir=None` (scenarios store no map files) — identical to today's "Scenarios → Load"; the step-1 note sets expectations.
- **Block-on-duplicate** in the wizard (vs. the Save flow's overwrite-confirm) is intentional: "create = new identity" should never silently overwrite.
- Out of scope: seed-value input, species-subset selection, a new preset store, build-from-scratch authoring, persisting map files in scenarios.
- **Imports at top:** tasks append code incrementally, but every `import` (core file AND test file — e.g. `import pytest`, the `osmose.demo`/`reader`/`scenarios` imports) must live at the TOP of its file, not mid-file, or `ruff check` fails with E402. Consolidate as you go.
