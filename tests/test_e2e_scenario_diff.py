"""End-to-end test for the Scenario Diff tab.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_scenario_diff.py -v -m e2e

Writes two synthetic runs into the real data/history/ directory (the app subprocess
reads that fixed path), each with a biomass CSV + a spatial NetCDF, then verifies the
biomass overlay and the three spatial maps render.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = Path(__file__).resolve().parent.parent
_HISTORY = _REPO / "data" / "history"
_SUBSTRATE = _REPO / "data" / "_scenario_diff_e2e"
_LOAD_TIMEOUT = 15_000


def _write_run(
    name: str, idx: int, *, base: float, config_snapshot: dict | None = None
) -> tuple[Path, Path]:
    """Create a synthetic output dir (biomass CSV + spatial NetCDF) and history record."""
    out = _SUBSTRATE / name
    out.mkdir(parents=True, exist_ok=True)
    # WIDE biomass CSV: Time + per-species columns (osm_biomass_*.csv convention)
    times = np.arange(10) / 1.0
    pd.DataFrame({"Time": times, "cod": base + times, "sprat": base + 0.5 * times}).to_csv(
        out / "osm_biomass_Simu0.csv", index=False
    )
    # Spatial NetCDF (time, species, lat, lon)
    ny, nx, nt = 4, 5, 10
    data = np.fromfunction(
        lambda t, s, y, x: base + s * 10 + t + y + x * 0.1, (nt, 2, ny, nx), dtype=float
    )
    xr.Dataset(
        {"spatial_biomass": (("time", "species", "lat", "lon"), data)},
        coords={
            "time": times,
            "species": ["cod", "sprat"],
            "lat": np.linspace(54.0, 55.0, ny),
            "lon": np.linspace(10.0, 12.0, nx),
        },
    ).to_netcdf(out / "osm_spatial_biomass_Simu0.nc")
    # History record (timestamp drives the selector label)
    ts = f"2026-06-13T0{idx}:00:00"
    rec = {
        "timestamp": ts,
        "config_snapshot": config_snapshot or {},
        "duration_sec": 1.0,
        "output_dir": str(out),
        "summary": {},
    }
    rec_path = _HISTORY / f"run_{ts.replace(':', '-')}.json"
    rec_path.write_text(json.dumps(rec))
    return rec_path, out


@pytest.fixture
def two_runs():
    _HISTORY.mkdir(parents=True, exist_ok=True)
    created = [_write_run("runA", 1, base=100.0), _write_run("runB", 2, base=130.0)]
    yield [ts for ts, _ in created]
    import shutil

    for rec_path, _ in created:
        rec_path.unlink(missing_ok=True)
    shutil.rmtree(_SUBSTRATE, ignore_errors=True)


@pytest.fixture
def two_runs_config():
    """Two runs whose config_snapshots differ: one changed, one A-only, one B-only key."""
    _HISTORY.mkdir(parents=True, exist_ok=True)
    cfg_a = {"predation.efficiency.sp0": "0.5", "mortality.natural.rate.sp0": "0.2"}
    cfg_b = {"predation.efficiency.sp0": "0.7", "movement.distance.sp0": "3"}
    created = [
        _write_run("runC", 3, base=100.0, config_snapshot=cfg_a),
        _write_run("runD", 4, base=130.0, config_snapshot=cfg_b),
    ]
    yield [ts for ts, _ in created]
    import shutil

    for rec_path, _ in created:
        rec_path.unlink(missing_ok=True)
    shutil.rmtree(_SUBSTRATE, ignore_errors=True)


def test_scenario_diff_renders_overlay_and_maps(page: Page, app: ShinyAppProc, two_runs):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)

    # Go to Results page, then the Scenario Diff tab.
    page.locator(".nav-pills .nav-link[data-value='results']").click()
    page.get_by_role("tab", name="Scenario Diff").click()

    # Wait for the history-backed selectors to populate (the effect runs after the
    # Results tab activates) before selecting — avoids a race on an empty <select>.
    # Assert OUR option exists (robust to other pre-existing runs in data/history).
    expect(page.locator("#diff_run_a option[value='2026-06-13T02:00:00']")).to_have_count(
        1, timeout=_LOAD_TIMEOUT
    )

    # Select baseline=runA (T01) and variant=runB (T02) by explicit timestamp value
    # (list_runs sorts DESC, so index order is not A-then-B).
    page.locator("#diff_run_a").select_option("2026-06-13T01:00:00")
    page.locator("#diff_run_b").select_option("2026-06-13T02:00:00")

    # Biomass overlay widget renders.
    expect(page.locator("#diff_biomass_chart")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # Three spatial map widgets render once both NetCDFs are open.
    expect(page.locator("#diff_map_delta")).to_be_visible(timeout=_LOAD_TIMEOUT)

    # Screenshot for manual confirmation (plotly content is in shadow DOM).
    page.screenshot(path=str(_REPO / "screenshots" / "scenario_diff_e2e.png"))


def test_scenario_diff_same_run_for_a_and_b(page: Page, app: ShinyAppProc, two_runs):
    """Selecting the SAME run for A and B must not crash (the identical-runs case).

    Exercises the real on-disk handle path: the server shares one open dataset rather
    than opening the same .nc twice (which would risk the HDF5-locking error). Unit
    tests only pass the same in-memory object, so this is the only coverage of it.
    """
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='results']").click()
    page.get_by_role("tab", name="Scenario Diff").click()
    expect(page.locator("#diff_run_a option[value='2026-06-13T01:00:00']")).to_have_count(
        1, timeout=_LOAD_TIMEOUT
    )

    # Same run for both sides → all-zero diff, single shared handle, no crash.
    page.locator("#diff_run_a").select_option("2026-06-13T01:00:00")
    page.locator("#diff_run_b").select_option("2026-06-13T01:00:00")

    expect(page.locator("#diff_biomass_chart")).to_be_visible(timeout=_LOAD_TIMEOUT)
    expect(page.locator("#diff_map_delta")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # The identical-runs caption should appear.
    expect(page.locator("#diff_biomass_caption")).to_contain_text(
        "Identical runs", timeout=_LOAD_TIMEOUT
    )


def test_scenario_diff_config_panel_shows_differences(
    page: Page, app: ShinyAppProc, two_runs_config
):
    """The config-diff panel lists changed, added, and removed keys for two runs."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='results']").click()
    page.get_by_role("tab", name="Scenario Diff").click()

    expect(page.locator("#diff_run_a option[value='2026-06-13T03:00:00']")).to_have_count(
        1, timeout=_LOAD_TIMEOUT
    )
    page.locator("#diff_run_a").select_option("2026-06-13T03:00:00")
    page.locator("#diff_run_b").select_option("2026-06-13T04:00:00")

    # Assert on CONTENT (the bare @render.ui div is zero-height until it recomputes, so
    # to_be_visible would flake). to_contain_text waits for the render to populate.
    cfg = page.locator("#diff_config_table")
    expect(cfg).to_contain_text("predation.efficiency.sp0", timeout=_LOAD_TIMEOUT)  # changed
    expect(cfg).to_contain_text("mortality.natural.rate.sp0", timeout=_LOAD_TIMEOUT)  # removed
    expect(cfg).to_contain_text("movement.distance.sp0", timeout=_LOAD_TIMEOUT)  # added

    page.screenshot(path=str(_REPO / "screenshots" / "scenario_diff_config_e2e.png"))
