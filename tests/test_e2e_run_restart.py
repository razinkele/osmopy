"""The Run page's restart-file input, end to end: an upload reaches the engine, and only when used.

Only warnings reach the run console, so "the run resumed" is not visible as text. The proof is a
control pair instead: a snapshot with a cell outside the grid FAILS the run with the engine's
"invalid restart file" (so the upload's key reached the engine), the same file with "Don't use
it" selected completes (so the mode gates it), and the valid snapshot completes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 60_000
_RUN_T = 180_000
_BALTIC = Path(__file__).resolve().parent.parent / "data" / "baltic" / "baltic_all-parameters.csv"


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> tuple[Path, Path]:
    """(valid, corrupt): the end-of-year-1 Baltic snapshot, and a copy with one cell off-grid."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = dict(OsmoseConfigReader().read(_BALTIC))
    cfg.update({"simulation.time.nyear": "1", "simulation.restart.enabled": "true"})
    out = tmp_path_factory.mktemp("restart_src")
    PythonEngine().run(cfg, out, seed=1)
    valid = next((out / "restart").glob("*.nc.0"))
    ds = xr.open_dataset(valid).load()
    x = ds["x"].values.copy()
    x[0] = 10_000.0
    ds["x"] = (ds["x"].dims, x, ds["x"].attrs)
    corrupt = out / "corrupt_snapshot.nc.0"
    ds.to_netcdf(corrupt)
    return valid, corrupt


def _run(page: Page, final: str) -> None:
    page.locator("#btn_run").click()
    expect(page.locator("#run_status")).to_contain_text("Running", timeout=_T)
    expect(page.locator("#run_status")).to_contain_text(final, timeout=_RUN_T)


def test_an_uploaded_restart_file_reaches_the_engine_only_when_used(
    page: Page, app: ShinyAppProc, snapshots
):
    valid, corrupt = snapshots
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_T)

    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#engineBtnPython").click()
    overrides = page.locator("#py_param_overrides")
    expect(overrides).to_be_visible(timeout=_T)
    overrides.fill("simulation.time.nyear=2")
    mode = page.locator("#restart_mode")
    expect(mode.locator("input[value='off']")).to_be_checked()

    # Choosing a file selects "resume"; a corrupt one then fails the run with the engine's reason.
    page.locator("#restart_upload").set_input_files(str(corrupt))
    expect(mode.locator("input[value='resume']")).to_be_checked(timeout=_T)
    _run(page, "Failed")
    expect(page.locator("#run_console")).to_contain_text("invalid restart file")

    # The same file, not used: the run is an ordinary one.
    mode.locator("input[value='off']").check()
    _run(page, "Complete")

    # A valid snapshot resumes.
    page.locator("#restart_upload").set_input_files(str(valid))
    expect(mode.locator("input[value='resume']")).to_be_checked(timeout=_T)
    _run(page, "Complete")
