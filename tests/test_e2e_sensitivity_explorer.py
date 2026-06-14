"""End-to-end test for the Parameter Sensitivity Explorer page.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_sensitivity_explorer.py -v -m e2e

Writes a synthetic Sobol artifact into the real data/history/sensitivity/ directory
(the app subprocess reads that fixed path), then verifies the tornado + table render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = Path(__file__).resolve().parent.parent
_SENS_DIR = _REPO / "data" / "history" / "sensitivity"
_TS = "2026-06-14T08:00:00"
_LOAD_TIMEOUT = 15_000


@pytest.fixture
def one_result():
    _SENS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "timestamp": _TS,
        "source": "test",
        "n_base": 16,
        "param_names": ["species.linf.sp0", "species.k.sp0", "predation.efficiency.sp0"],
        "param_bounds": [[20.0, 60.0], [0.1, 0.5], [0.3, 0.9]],
        "objective_names": ["Biomass RMSE"],
        "n_objectives": 1,
        "S1": [0.40, 0.10, 0.25],
        "ST": [0.50, 0.15, 0.30],
        "S1_conf": [0.05, 0.02, 0.03],
        "ST_conf": [0.06, 0.03, 0.04],
    }
    path = _SENS_DIR / f"sobol_{_TS.replace(':', '-')}.json"
    path.write_text(json.dumps(artifact))
    yield _TS
    path.unlink(missing_ok=True)


def test_sensitivity_explorer_renders(page: Page, app: ShinyAppProc, one_result):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    # Open the Sensitivity page (triggers _populate_runs).
    page.locator(".nav-pills .nav-link[data-value='sensitivity']").click()

    # The history-backed selector populates after the page activates.
    expect(page.locator(f"#sens_run option[value='{_TS}']")).to_have_count(1, timeout=_LOAD_TIMEOUT)
    page.locator("#sens_run").select_option(_TS)

    # Tornado widget renders.
    expect(page.locator("#sens_tornado")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # Table renders the ranked params (content assertion — bare @render.ui is zero-height
    # until it recomputes, so to_contain_text waits for population).
    expect(page.locator("#sens_table")).to_contain_text("species.linf.sp0", timeout=_LOAD_TIMEOUT)
    # Export buttons present.
    expect(page.locator("#sens_export_csv")).to_be_visible(timeout=_LOAD_TIMEOUT)

    page.screenshot(path=str(_REPO / "screenshots" / "sensitivity_explorer_e2e.png"))
