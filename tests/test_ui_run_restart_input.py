"""The Run page's restart-file input: which keys an upload sets (ui.pages.run.restart_run_keys)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ui.pages.run import RESTART_MODES, restart_run_keys


@pytest.fixture
def upload(tmp_path: Path) -> list[dict]:
    src = tmp_path / "uploaded" / "0.nc"  # Shiny stores uploads under a temp name
    src.parent.mkdir()
    src.write_bytes(b"snapshot bytes")
    return [{"name": "baltic_snapshot_step23.nc.0", "datapath": str(src)}]


def test_resume_sets_the_restart_key_and_blanks_the_init_key(upload, tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    keys = restart_run_keys("resume", upload, work)
    assert keys == {
        "simulation.restart.file": str(work / "restart_input.nc"),
        "population.initialization.file": "",
    }
    assert (work / "restart_input.nc").read_bytes() == b"snapshot bytes"


def test_init_blanks_the_restart_key_which_would_otherwise_win(upload, tmp_path):
    keys = restart_run_keys("init", upload, tmp_path)
    assert keys == {
        "population.initialization.file": str(tmp_path / "restart_input.nc"),
        "simulation.restart.file": "",
    }


@pytest.mark.parametrize(
    ("mode", "has_upload"), [("off", True), ("resume", False), ("init", False)]
)
def test_nothing_changes_without_a_file_or_when_off(upload, tmp_path, mode, has_upload):
    assert restart_run_keys(mode, upload if has_upload else None, tmp_path) == {}
    assert not (tmp_path / "restart_input.nc").exists()


def test_the_modes_offered_are_the_ones_handled():
    assert set(RESTART_MODES) == {"off", "resume", "init"}
