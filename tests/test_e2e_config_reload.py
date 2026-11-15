"""Loading a second example must give that example's config, whatever tabs were viewed.

A tab hidden during a load does not re-render (Shiny suspends hidden outputs), so its inputs keep
the FIRST config's values; its sync effect still fires on the load and wrote them into the second.
Measured 2026-11-15 before the fix: load EEC full, view Setup, load Baltic -> 169 of Baltic's 700
keys were EEC full's (species names, growth, maturity, recruitment, ...); via Movement, 92 keys
(map species, ages, steps). The fix (AppState.mark_rendered / rendered_current) blocks the write
until the tab has re-rendered for the current load.

The other arm matters as much: a sync that never ran would pass the first test, so an edit made
after the reload must still be saved.

The instrument is the saved config (a scenario), checked two ways against the example's own file:
every key the file sets keeps its value, and no extra key carries the FIRST example's value. Extra
keys are otherwise allowed: rendering a tab writes the schema default for keys the example leaves
unset (pre-existing, and inert here: a config saved after these sequences runs bit-identically to
the file over 2 years, measured 2026-11-15).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_config_reload.py -m e2e -o addopts="" \
        -p no:cacheprovider
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import list_demos, osmose_demo
from osmose.schema import build_registry
from osmose.schema.base import ParamType
from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 30_000
_SCENARIOS_DIR = Path("data/scenarios")
_IGNORED = {"_osmose.config.dir"}  # each load extracts the demo to a new temp dir


def _start(page: Page, app: ShinyAppProc) -> None:
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)


def _load_example(page: Page, example: str) -> None:
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_T)
    page.select_option("#load_example", example)
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_T)
    page.wait_for_timeout(3_000)


def _visit(page: Page, tab: str) -> None:
    page.locator(f".nav-pills .nav-link[data-value='{tab}']").click()
    page.wait_for_timeout(4_000)  # render, then let the sync effects run


def _save_scenario(page: Page, name: str) -> dict[str, str]:
    page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
    page.locator("#scenario_name").fill(name, timeout=_T)
    page.click("#btn_save_scenario")
    path = _SCENARIOS_DIR / name / "scenario.json"
    for _ in range(40):
        if path.exists():
            break
        page.wait_for_timeout(250)
    return json.loads(path.read_text())["config"]


def _expected(example: str, tmp_path: Path) -> dict[str, str]:
    info = osmose_demo(example, tmp_path)
    return dict(OsmoseConfigReader().read(Path(info["config_file"])))


def _differences(expected: dict[str, str], saved: dict[str, str]) -> dict[str, tuple]:
    return {
        k: (v, saved.get(k)) for k, v in expected.items() if k not in _IGNORED and saved.get(k) != v
    }


_REGISTRY = build_registry()


def _is_rendered_default(key: str, value: str) -> bool:
    """The value a rendered input writes for a key the config leaves unset (its schema default;
    a numeric field without one renders 0)."""
    field = _REGISTRY.match_field(key)
    if field is None:
        return False
    if field.param_type in (ParamType.FLOAT, ParamType.INT):
        try:
            return float(value) == float(field.default if field.default is not None else 0)
        except (TypeError, ValueError):
            return False
    if field.param_type == ParamType.BOOL:
        return value.lower() == str(bool(field.default)).lower()
    return value == ("" if field.default is None else str(field.default))


def _leaked(first: dict[str, str], expected: dict[str, str], saved: dict[str, str]) -> list[str]:
    """Saved-only keys that carry the first example's value (and are not a rendered default)."""
    extras = set(saved) - set(expected) - _IGNORED
    return sorted(
        k
        for k in extras
        if k in first and saved[k] == first[k] and not _is_rendered_default(k, saved[k])
    )


@pytest.mark.parametrize("tab", ["setup", "movement", "fishing", "forcing", "grid"])
@pytest.mark.parametrize(("first", "second"), [("eec_full", "baltic"), ("baltic", "eec_full")])
def test_second_load_is_not_corrupted_by_a_viewed_tab(
    page: Page, app: ShinyAppProc, tmp_path, tab, first, second
):
    expected = _expected(second, tmp_path / "second")
    first_cfg = _expected(first, tmp_path / "first")
    name = f"e2e_reload_{tab}_{uuid.uuid4().hex[:8]}"
    try:
        _start(page, app)
        _load_example(page, first)
        _visit(page, tab)
        _load_example(page, second)
        _visit(page, tab)  # re-render for the second config; nothing may change
        saved = _save_scenario(page, name)
        wrong = _differences(expected, saved)
        assert not wrong, (
            f"{first} -> {tab} -> {second}: {len(wrong)} keys differ from {second}'s file, "
            f"e.g. {dict(list(wrong.items())[:4])}"
        )
        leaked = _leaked(first_cfg, expected, saved)
        assert not leaked, f"{first} -> {tab} -> {second}: {first}'s values leaked: {leaked[:6]}"
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)


@pytest.mark.parametrize(
    ("tab", "input_id", "key", "value"),
    [
        ("setup", "spt_species_linf_0", "species.linf.sp0", "77.0"),
        ("movement", "movement_initialage_map0", "movement.initialage.map0", "7"),
        # Grid and Forcing edits were never saved at all before 2026-11-15: the sync aborted on an
        # unrendered advanced key (SilentException) before writing anything.
        ("forcing", "ltl_nstep", "ltl.nstep", "13"),
        ("grid", "grid_nlon", "grid.nlon", "51"),
    ],
)
def test_an_edit_after_the_reload_is_still_saved(
    page: Page, app: ShinyAppProc, tmp_path, tab, input_id, key, value
):
    expected = _expected("baltic", tmp_path / "second")
    first_cfg = _expected("eec_full", tmp_path / "first")
    name = f"e2e_reload_edit_{tab}_{uuid.uuid4().hex[:8]}"
    try:
        _start(page, app)
        _load_example(page, "eec_full")
        _visit(page, tab)
        _load_example(page, "baltic")
        _visit(page, tab)
        page.locator(f"#{input_id}").fill(value, timeout=_T)
        page.locator(f"#{input_id}").press("Tab")
        page.wait_for_timeout(2_000)
        saved = _save_scenario(page, name)
        assert float(saved[key]) == float(value), f"edit lost: {key} = {saved.get(key)!r}"
        others = {k: v for k, v in saved.items() if k != key}
        wrong = _differences({k: v for k, v in expected.items() if k != key}, others)
        assert not wrong, f"{len(wrong)} other keys changed, e.g. {dict(list(wrong.items())[:4])}"
        leaked = _leaked(first_cfg, {k: v for k, v in expected.items() if k != key}, others)
        assert not leaked, f"EEC full's values leaked: {leaked[:6]}"
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)


def test_the_resource_copy_button_does_not_copy_stale_species(
    page: Page, app: ShinyAppProc, tmp_path
):
    """Both tables' "Copy to all" buttons once set the same input, so the resource table's button
    also ran the species copy, which read the species table's stale inputs (EEC full's species 0)
    and wrote them over Baltic's species 1..n. Each table now has its own event, gated by its render."""
    from osmose.schema.species import SPECIES_FIELDS

    expected = _expected("baltic", tmp_path)
    n_focal = int(expected["simulation.nspecies"])
    species_keys = {
        f.resolve_key(i) for f in SPECIES_FIELDS if f.indexed for i in range(n_focal)
    } & set(expected)
    name = f"e2e_reload_copy_{uuid.uuid4().hex[:8]}"
    try:
        _start(page, app)
        _load_example(page, "eec_full")
        _visit(page, "setup")
        _load_example(page, "baltic")
        _visit(page, "forcing")
        page.locator("#resource_panels button", has_text="Copy to all").click(timeout=_T)
        page.wait_for_timeout(2_000)
        saved = _save_scenario(page, name)
        wrong = {
            k: (expected[k], saved.get(k)) for k in species_keys if saved.get(k) != expected[k]
        }
        assert not wrong, (
            f"species values changed by the resource copy: {dict(list(wrong.items())[:4])}"
        )
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)


@pytest.mark.parametrize("demo", list_demos())
def test_a_plain_load_gives_exactly_the_file(page: Page, app: ShinyAppProc, tmp_path, demo):
    """Loading happens on the Grid page, which is visible during the load, so its sync runs then.
    Once that sync stopped aborting it wrote the Grid fields' defaults into every loaded config
    (e.g. grid.java.classname = OriginalGrid). A plain load must give exactly the file: no
    changed key and no added key."""
    expected = _expected(demo, tmp_path)
    name = f"e2e_plain_{demo}_{uuid.uuid4().hex[:8]}"
    try:
        _start(page, app)
        _load_example(page, demo)
        saved = _save_scenario(page, name)
        added = sorted(set(saved) - set(expected) - _IGNORED)
        assert not _differences(expected, saved), f"{demo}: changed {_differences(expected, saved)}"
        assert not added, f"{demo}: the load added keys {added}"
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)
