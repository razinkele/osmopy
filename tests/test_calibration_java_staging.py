"""Calibration's Java path must stage the base config for the jar, like every other Java run.

It handed Java the caller's base config as given and translated only the candidate overrides.
Neither jar loads the bundled native-4.4.x configs raw, so with the Bay of Biscay config every
candidate exited 1 and the calibration aborted ("3/3 candidates failed"), seen 2026-11-07 on the
4.4.1 jar. Its failure log was no help either: it showed only stderr, which holds the SLF4J
banner, while Java prints its exception to stdout.

These need no jar: ``subprocess.run`` is faked and the staged config is inspected directly.
``tests/test_calibration_problem_python_engine.py::test_objective_values_match_between_engines``
runs real jars when ``OSMOSE_JAR`` is set.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from osmose import java_staging
from osmose.calibration import FreeParameter, OsmoseCalibrationProblem

ROOT = Path(__file__).resolve().parents[1]
BOB = ROOT / "data" / "examples" / "osm_all-parameters.csv"
JAR_NAMES = {
    "4.4.1": "osmose-4.4.1-jar-with-dependencies.jar",
    "4.3.3": "osmose_4.3.3-jar-with-dependencies.jar",
}
# What each jar version must find in its staged master (and the raw config lacks).
_REQUIRED = {"4.4.1": "species.biomass.file.sp", "4.3.3": "species.biomass.nsteps.year"}


def _problem(tmp_path: Path, base: Path = BOB, version: str = "4.4.1") -> OsmoseCalibrationProblem:
    return OsmoseCalibrationProblem(
        free_params=[FreeParameter("mortality.fishing.rate.sp0", 0.1, 0.5)],
        objective_fns=[lambda r: 1.0],
        base_config_path=base,
        jar_path=tmp_path / JAR_NAMES[version],
        work_dir=tmp_path / "work",
        use_java_engine=True,
        enable_cache=False,
    )


def _fake_java(monkeypatch, returncode=1, stdout=b"", stderr=b"") -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _master(argv: list[str]) -> Path:
    return Path(argv[argv.index("-jar") + 2])


def _keys(master: Path) -> set[str]:
    return {ln.partition(";")[0].strip() for ln in master.read_text().splitlines() if ";" in ln}


@pytest.mark.parametrize("version", ["4.4.1", "4.3.3"])
def test_java_gets_the_base_config_staged_for_the_jar(version, tmp_path, monkeypatch):
    calls = _fake_java(monkeypatch)
    _problem(tmp_path, version=version)._run_java_subprocess(
        {"mortality.fishing.rate.sp0": "0.3"}, run_id=0
    )
    master = _master(calls[0])
    assert master != BOB, "Java was handed the raw base config"
    assert any(k.startswith(_REQUIRED[version]) for k in _keys(master)), (
        f"staged master for {version} lacks {_REQUIRED[version]}*; that jar exits without it"
    )
    assert any(a.startswith("-P") and a.endswith("=0.3") for a in calls[0]), "candidate dropped"


def test_the_base_config_is_staged_once_not_per_candidate(tmp_path, monkeypatch):
    calls = _fake_java(monkeypatch)
    n_staged = 0
    real = java_staging.stage_config_for_java

    def counting(*args, **kwargs):
        nonlocal n_staged
        n_staged += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(java_staging, "stage_config_for_java", counting)
    problem = _problem(tmp_path)
    problem._run_java_subprocess({}, run_id=0)
    problem._run_java_subprocess({}, run_id=1)
    assert n_staged == 1
    assert _master(calls[0]) == _master(calls[1])


def test_staging_overrides_reach_java_and_win(tmp_path, monkeypatch):
    """As in the Run tab: a staging override (e.g. the background cutoff fix) beats a candidate."""
    calls = _fake_java(monkeypatch)
    staged = tmp_path / "staged.csv"
    staged.write_text("simulation.nspecies ; 1\n")
    monkeypatch.setattr(
        java_staging,
        "stage_config_for_java",
        lambda *a, **k: (staged, {"output.cutoff.enabled": "false"}),
    )
    _problem(tmp_path)._run_java_subprocess({"output.cutoff.enabled": "true"}, run_id=0)
    argv = calls[0]
    assert _master(argv) == staged
    assert "-Poutput.cutoff.enabled=false" in argv
    assert "-Poutput.cutoff.enabled=true" not in argv


def test_a_staging_failure_scores_the_run_as_failed(tmp_path, monkeypatch, caplog):
    calls = _fake_java(monkeypatch, returncode=0)
    with caplog.at_level(logging.WARNING):
        result = _problem(tmp_path, base=tmp_path / "missing.csv")._run_java_subprocess({}, 0)
    assert result is None
    assert not calls, "Java ran without a staged config"
    assert "missing.csv" in caplog.text


def test_the_failure_log_carries_javas_stdout(tmp_path, monkeypatch, caplog):
    severe = b"osmose[severe] NETCDF_BIOMASS resource forcing is used but parameters are missing\n"
    _fake_java(monkeypatch, returncode=1, stdout=severe, stderr=b"SLF4J: Failed to load class\n")
    with caplog.at_level(logging.WARNING):
        result = _problem(tmp_path)._run_java_subprocess({}, run_id=0)
    assert result is None
    assert "NETCDF_BIOMASS" in caplog.text, "the log showed only stderr's SLF4J banner"
    assert (tmp_path / "work" / "run_0" / "stdout.txt").read_bytes() == severe
