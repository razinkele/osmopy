# Real-time Config Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show config validation problems live on the Setup page as the user edits, instead of only at Run time.

**Architecture:** Extract the existing Run-gate validation sequence into one pure helper `summarize_config_validation` (the single source of truth), then consume it from all three callers — the Run gate, the new live Setup panel, and the `osmose validate` CLI. No new validation logic; the panel is informational, the Run gate keeps its blocking authority.

**Tech Stack:** Python 3.12, Shiny for Python (reactive `@reactive.calc` / `@render.ui`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-06-08-realtime-config-validation-design.md` (2 in-loop review rounds folded).

**Branch:** `feature/realtime-config-validation` (already checked out; round-2 spec committed at `79acb47`).

---

## File Structure

- **Modify** `osmose/config/validator.py` — add `summarize_config_validation` helper (already imports `Path` and `ParameterRegistry`, no new imports).
- **Modify** `ui/pages/run.py` — replace the inline 6-line gate (`:478-485`) with the helper; prune now-unused imports (`:10-14`).
- **Modify** `osmose/cli.py` — replace the inline composition in `cmd_validate` (`:29-33`) with the helper; prune imports (`:18-22`).
- **Modify** `ui/pages/setup.py` — add `@reactive.calc _config_validation` + `@render.ui config_validation` + `ui.output_ui("config_validation")` in `setup_ui` (with the 10-line cap, message wrapping, `aria-live`).
- **Modify** `tests/test_validator.py` — unit tests for the helper + DRY-lock test.
- **Modify** `tests/test_wiring.py` (create if absent) — source-string wiring asserts.
- **Modify** `CHANGELOG.md` — `## [Unreleased]` → `### Added` note.

Run all commands from the repo root `/home/razinka/osmose/osmose-python`. Use `.venv/bin/python -m pytest` and `.venv/bin/ruff` (never bare `python`/`pytest`/`ruff`).

---

### Task 1: `summarize_config_validation` helper + unit tests + DRY-lock test

**Files:**
- Modify: `osmose/config/validator.py` (append after `check_species_consistency`, ~`:148`)
- Test: `tests/test_validator.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validator.py`:

```python
def test_summarize_clean_config_returns_empty():
    from osmose.config.validator import summarize_config_validation
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"simulation.nspecies": "1", "species.name.sp0": "cod", "species.type.sp0": "focal"}
    errors, warnings = summarize_config_validation(config, registry)
    assert errors == []
    assert warnings == []


def test_summarize_reports_enum_error():
    from osmose.config.validator import summarize_config_validation
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"species.type.sp0": "invalid_type"}
    errors, _ = summarize_config_validation(config, registry)
    assert any("invalid_type" in e for e in errors)


def test_summarize_reports_missing_species_warning():
    from osmose.config.validator import summarize_config_validation
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"simulation.nspecies": "2"}  # no species.name.sp0/sp1
    _, warnings = summarize_config_validation(config, registry)
    assert len([w for w in warnings if "species name" in w.lower()]) == 2


def test_summarize_skips_file_checks_when_no_dir():
    from osmose.config.validator import summarize_config_validation
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"movement.file.map0": "does_not_exist.csv"}
    errors, _ = summarize_config_validation(config, registry, config_dir=None)
    assert not any("File not found" in e for e in errors)


def test_summarize_reports_file_error_with_dir(tmp_path):
    from osmose.config.validator import summarize_config_validation
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"movement.file.map0": "missing_map.csv"}
    errors, _ = summarize_config_validation(config, registry, config_dir=tmp_path)
    assert any("File not found" in e and "movement.file.map0" in e for e in errors)


def test_summarize_does_not_raise_on_malformed():
    from osmose.config.validator import summarize_config_validation
    from osmose.schema import build_registry

    registry = build_registry()
    config = {"species.type.sp0": "invalid_type", "simulation.nspecies": "2"}
    errors, warnings = summarize_config_validation(config, registry)  # must not raise
    assert isinstance(errors, list) and isinstance(warnings, list)


def test_summarize_matches_inline_composition_dry_lock(tmp_path):
    """The helper must be byte-identical to the old inline gate sequence."""
    from osmose.config.validator import (
        summarize_config_validation,
        validate_config,
        check_file_references,
        check_species_consistency,
    )
    from osmose.schema import build_registry

    registry = build_registry()
    config = {
        "simulation.nspecies": "2",
        "species.type.sp0": "invalid_type",
        "movement.file.map0": "missing_map.csv",
    }
    for config_dir in (None, tmp_path):
        # Old inline composition (run.py:479-485 / cli.py:29-33)
        inline_err, inline_warn = validate_config(config, registry)
        if config_dir:
            inline_err.extend(check_file_references(config, str(config_dir), registry))
        inline_warn.extend(check_species_consistency(config))
        # Helper
        helper_err, helper_warn = summarize_config_validation(config, registry, config_dir)
        assert helper_err == inline_err
        assert helper_warn == inline_warn
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_validator.py -k summarize -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_config_validation'`.

- [ ] **Step 3: Write the helper**

Append to `osmose/config/validator.py` (after `check_species_consistency`, end of file). No new imports — `Path` and `ParameterRegistry` are already imported at `:5` and `:8`:

```python
def summarize_config_validation(
    config: dict[str, str],
    registry: ParameterRegistry,
    config_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Aggregate all config checks into (errors, warnings) of human-readable strings.

    The single source of truth for "what's wrong with this config", consumed by
    the Setup live panel, the Run gate, and the `osmose validate` CLI. Mirrors the
    former inline Run-gate sequence exactly: type/range/enum checks, then file-reference
    checks (only when config_dir is set — relative paths would false-error otherwise),
    then species-name consistency (warnings). Pure; never raises on a partial config.

    The returned lists are fresh (validate_config builds new local lists), so callers
    may safely keep extending them.
    """
    errors, warnings = validate_config(config, registry)
    if config_dir:  # truthiness — matches the gate's `if source_dir:` exactly
        errors.extend(check_file_references(config, str(config_dir), registry))
    warnings.extend(check_species_consistency(config))
    return errors, warnings
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_validator.py -k summarize -v`
Expected: PASS (7 tests). If `test_summarize_*file*` fail, confirm `movement.file.map0` resolves to a `FILE_PATH` field via `build_registry().match_field("movement.file.map0").param_type` — if it isn't FILE_PATH, swap to `species.file.sp0` in both file tests and the DRY-lock test.

- [ ] **Step 5: Commit**

```bash
git add osmose/config/validator.py tests/test_validator.py
git commit -m "feat(config): add summarize_config_validation helper (single source of truth)"
```

---

### Task 2: DRY-refactor the Run gate and the CLI onto the helper

**Files:**
- Modify: `ui/pages/run.py:10-14` (imports), `ui/pages/run.py:478-485` (gate)
- Modify: `osmose/cli.py:18-22` (imports), `osmose/cli.py:29-33` (composition)
- Test: `tests/test_wiring.py` (create), existing `tests/test_cli.py` (unchanged, must still pass)

- [ ] **Step 1: Write the failing wiring test**

Create `tests/test_wiring.py` (if it already exists, append these functions):

```python
"""Source-string wiring asserts for the config-validation feature.

validator.py has no __all__, so these target the source text directly — the
correct mechanism for asserting imports were pruned and the helper is used.
"""
from pathlib import Path


def test_run_gate_uses_helper_and_prunes_imports():
    src = Path("ui/pages/run.py").read_text()
    assert "summarize_config_validation" in src
    # Old inline validators must be fully pruned (ruff F401 otherwise).
    assert "check_file_references" not in src
    assert "check_species_consistency" not in src
    assert "validate_config" not in src  # not a substring of summarize_config_validation


def test_cli_uses_helper_and_prunes_imports():
    src = Path("osmose/cli.py").read_text()
    assert "summarize_config_validation" in src
    assert "check_file_references" not in src
    assert "check_species_consistency" not in src
    assert "validate_config" not in src
```

- [ ] **Step 2: Run the wiring test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wiring.py -v`
Expected: FAIL — `run.py`/`cli.py` still import and use the old names.

- [ ] **Step 3: Refactor `ui/pages/run.py`**

Replace the import block at `ui/pages/run.py:10-14`:

```python
from osmose.config.validator import (
    check_file_references,
    check_species_consistency,
    validate_config,
)
```

with:

```python
from osmose.config.validator import summarize_config_validation
```

Replace the gate sequence at `ui/pages/run.py:478-485`:

```python
        config = state.config.get()
        errors, warnings = validate_config(config, state.registry)
        source_dir = state.config_dir.get()
        if source_dir:
            file_errors = check_file_references(config, str(source_dir), state.registry)
            errors.extend(file_errors)
        species_warnings = check_species_consistency(config)
        warnings.extend(species_warnings)
```

with (keep `config = state.config.get()` — it's still passed to the engine runners below):

```python
        config = state.config.get()
        errors, warnings = summarize_config_validation(
            config, state.registry, state.config_dir.get()
        )
```

The downstream `if errors:` / `if warnings:` blocking blocks (`:487-502`) are unchanged.

- [ ] **Step 4: Refactor `osmose/cli.py`**

Replace the import block at `osmose/cli.py:18-22`:

```python
    from osmose.config.validator import (
        validate_config,
        check_file_references,
        check_species_consistency,
    )
```

with:

```python
    from osmose.config.validator import summarize_config_validation
```

Replace the composition at `osmose/cli.py:29-33`:

```python
    errors, warnings = validate_config(config, registry)
    file_errors = check_file_references(config, str(config_path.parent), registry)
    errors.extend(file_errors)
    species_warnings = check_species_consistency(config)
    warnings.extend(species_warnings)
```

with (a CLI always reads from a file, so `config_path.parent` is always truthy → file checks always run, exactly as before):

```python
    errors, warnings = summarize_config_validation(config, registry, config_path.parent)
```

- [ ] **Step 5: Run the wiring test + the existing run/cli tests + the DRY-lock test**

Run: `.venv/bin/python -m pytest tests/test_wiring.py tests/test_cli.py tests/test_validator.py -v`
Expected: PASS. (`test_cli_validate_*` prove the CLI behavior is preserved; the DRY-lock test from Task 1 proves the gate sequence is byte-identical.)

- [ ] **Step 6: Lint to confirm no F401**

Run: `.venv/bin/ruff check ui/pages/run.py osmose/cli.py`
Expected: no output (clean — no unused-import errors).

- [ ] **Step 7: Commit**

```bash
git add ui/pages/run.py osmose/cli.py tests/test_wiring.py
git commit -m "refactor(config): route Run gate and CLI through summarize_config_validation (DRY)"
```

---

### Task 3: Live validation panel on the Setup page

**Files:**
- Modify: `ui/pages/setup.py` (imports `:3-15`, `setup_ui` `:23-43`, `setup_server` body)
- Test: `tests/test_wiring.py` (append a setup wiring assert)

- [ ] **Step 1: Write the failing wiring test**

Append to `tests/test_wiring.py`:

```python
def test_setup_wires_live_validation_panel():
    src = Path("ui/pages/setup.py").read_text()
    assert "summarize_config_validation" in src
    assert 'ui.output_ui("config_validation")' in src
    assert "aria-live" in src
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wiring.py::test_setup_wires_live_validation_panel -v`
Expected: FAIL — setup.py has no panel yet.

- [ ] **Step 3: Add the import**

In `ui/pages/setup.py`, after the existing `from osmose.schema.species import SPECIES_FIELDS` (`:8`), add:

```python
from osmose.config.validator import summarize_config_validation
```

- [ ] **Step 4: Insert the output slot in `setup_ui`**

In `ui/pages/setup.py::setup_ui` (`:24-43`), insert `ui.output_ui("config_validation")` as the **second child** of the outer `ui.div` — between `expand_tab(...)` and `ui.layout_columns(...)`:

```python
def setup_ui():
    return ui.div(
        expand_tab("Simulation Settings", "setup"),
        ui.output_ui("config_validation"),
        ui.layout_columns(
            # Left column: Simulation settings
            ui.card(
                collapsible_card_header("Simulation Settings", "setup"),
                ui.output_ui("simulation_fields"),
            ),
            # Right column: Species configuration (dynamic)
            ui.card(
                ui.card_header("Species Configuration"),
                ui.input_numeric("n_species", "Number of focal species", value=3, min=1, max=20),
                ui.input_switch("show_advanced_species", "Show advanced parameters", value=False),
                ui.output_ui("species_panels"),
            ),
            col_widths=[4, 8],
        ),
        class_="osm-split-layout",
        id="split_setup",
    )
```

- [ ] **Step 5: Add the calc + render to `setup_server`**

In `ui/pages/setup.py::setup_server`, add these two functions (place them right after the `def setup_server(...):` line, before `simulation_fields`):

```python
    @reactive.calc
    def _config_validation():
        """(loaded, errors, warnings). Recomputes on every edit AND on load,
        since every loader sets state.config directly. Never raises — a half-entered
        config must never crash the Setup tab."""
        config = state.config.get()
        config_dir = state.config_dir.get()
        loaded = bool(config) and "simulation.nspecies" in config
        if not loaded:
            return (False, [], [])
        try:
            errors, warnings = summarize_config_validation(config, state.registry, config_dir)
        except Exception as exc:  # noqa: BLE001 - panel must degrade, never crash the tab
            return (True, [], [f"validation unavailable: {exc}"])
        return (True, errors, warnings)

    @render.ui
    def config_validation():
        loaded, errors, warnings = _config_validation()

        def _issue_lines(items, css, cap=10):
            lines = [
                ui.div(f"• {m}", class_=css, style="overflow-wrap:anywhere")
                for m in items[:cap]
            ]
            if len(items) > cap:
                lines.append(
                    ui.div(f"… and {len(items) - cap} more", class_="small text-muted")
                )
            return lines

        if not loaded:
            inner = ui.div("No configuration loaded", class_="small text-muted")
        elif not errors and not warnings:
            inner = ui.div("✓ Configuration valid", class_="small text-success")
        else:
            badge_css = "fw-bold text-danger" if errors else "fw-bold text-warning"
            inner = ui.div(
                ui.div(
                    f"{len(errors)} error(s) · {len(warnings)} warning(s)",
                    class_=badge_css,
                ),
                *_issue_lines(errors, "small text-danger"),
                *_issue_lines(warnings, "small text-warning"),
            )

        return ui.card(
            ui.div(inner, **{"aria-live": "polite", "aria-atomic": "true"}),
            class_="mb-2",
        )
```

- [ ] **Step 6: Run the wiring test + a page-build smoke check**

Run: `.venv/bin/python -m pytest tests/test_wiring.py -v`
Expected: PASS (all three wiring asserts).

Run a build smoke check that `setup_ui()` constructs without error:

```bash
.venv/bin/python -c "import ui.pages.setup as s; s.setup_ui(); print('setup_ui builds OK')"
```
Expected: `setup_ui builds OK`.

- [ ] **Step 7: Lint**

Run: `.venv/bin/ruff check ui/pages/setup.py && .venv/bin/ruff format --check ui/pages/setup.py`
Expected: clean (no output). If format-check fails, run `.venv/bin/ruff format ui/pages/setup.py` and re-stage.

- [ ] **Step 8: Commit**

```bash
git add ui/pages/setup.py tests/test_wiring.py
git commit -m "feat(ui): live config validation panel on the Setup page"
```

---

### Task 4: CHANGELOG + full-suite verification

**Files:**
- Modify: `CHANGELOG.md` (under `## [Unreleased]` → `### Added`, as the first bullet)

- [ ] **Step 1: Add the CHANGELOG note**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added` (insert as the first bullet of that list, before the existing `**config (parser diagnostics):**` bullet):

```markdown
- **ui (config validation):** the Setup page now shows a live validation summary panel
  (type/range/enum errors, missing species names, missing file references) as you edit,
  mirroring the Run gate via a new `summarize_config_validation` helper that the Run gate
  and the `osmose validate` CLI now also use. The panel caps long issue lists and announces
  updates via `aria-live`. Informational only — the Run gate keeps its blocking authority.
```

- [ ] **Step 2: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, with the new tests added (no regressions). Note the pass count.

- [ ] **Step 3: Full-repo lint + format check (matches CI)**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: clean (no output). CI runs both `ruff check` AND `ruff format --check` on these three dirs — a green check can sit next to a red format, so run both.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note live config validation panel"
```

- [ ] **Step 5: Manual Playwright run-through (render fn not unit-tested, per convention)**

Launch: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000` and drive with Playwright:
- Before any load → panel shows neutral **"No configuration loaded"** (grey, not green).
- Load a config (Grid example) → panel shows **"✓ Configuration valid"** or the real issues.
- Edit a numeric field out of range → the error appears **live**; fix it → returns to green.
- Toggle **"Show advanced parameters"** → no errors appear for keys with no visible input.
- Load a config whose files were moved (or one with broken file refs) → confirm the errors group **caps at 10** with a `… and N more` tail (not a 50-line wall) and a long File-not-found **path wraps, not clips**.
- Confirm the card **spans full width above** the columns. If it collapses into a flex column, add `w-100` to the card's `class_` (→ `class_="mb-2 w-100"`) and re-verify.
- Confirm the card carries `aria-live="polite"` (inspect the DOM).
- Confirm rapid species-table typing does not cause input focus loss (the panel re-renders independently).
- Confirm the existing Run gate still blocks on the same errors.

---

## Self-Review (completed during authoring)

- **Spec coverage:** helper (Task 1) ✓; Run-gate refactor + import prune (Task 2) ✓; CLI refactor — the round-2 "third copy" fold (Task 2) ✓; live panel calc+render+output_ui with cap/wrap/aria-live (Task 3) ✓; tests — unit + DRY-lock + wiring (Tasks 1-3) ✓; CHANGELOG (Task 4) ✓; manual Playwright (Task 4) ✓. The `loaded`-flag false-green fix lives in the calc (Task 3 Step 5) ✓.
- **Placeholder scan:** none — every code step shows complete code; every command shows expected output.
- **Type consistency:** `summarize_config_validation(config, registry, config_dir)` signature identical across helper definition (Task 1), run.py call (Task 2), cli.py call (Task 2), and the calc (Task 3). Return shape `(errors, warnings)` consistent; calc wraps it as `(loaded, errors, warnings)` consistently between calc and render.
- **Known fallback:** if `movement.file.map0` is not a `FILE_PATH` field in the registry, Task 1 Step 4 directs swapping to `species.file.sp0` in the file tests (and the DRY-lock test).
