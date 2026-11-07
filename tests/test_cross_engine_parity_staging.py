"""``scripts/cross_engine_parity_440.py`` must stage its Java arms like every other Java run.

It staged with ``write_temp_config`` plus its own 4.3.3 step-count injection, bypassing
``osmose.java_staging.stage_config_for_java``. For the configs it has been run on, Bay of
Biscay and EEC, the two produce identical masters (checked 2026-11-07). But on a config with
background species it skipped background staging and the 4.4.x name reconcile, so a Baltic
4.4.1 arm would have hit GitHub #138 (names like ``cod_west`` that 4.4.1 cannot resolve) and
lacked the background-staging ``-P`` overrides.

``java_rep`` is replaced by a recorder, so these need no jar and run in CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "xeng_staging", ROOT / "scripts" / "cross_engine_parity_440.py"
)
assert _spec and _spec.loader
xeng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xeng)

BALTIC = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
BOB = ROOT / "data" / "examples" / "osm_all-parameters.csv"


def _record_java_reps(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_java_rep(ver, master, years, odir, spinup, prefix, overrides=None):
        calls.append({"ver": ver, "master": Path(master), "overrides": overrides})
        return {m: {"sp": 1.0} for m in xeng.METRICS + xeng.OPTIONAL_METRICS}

    monkeypatch.setattr(xeng, "java_rep", fake_java_rep)
    return calls


def _entries(master: Path) -> list[tuple[str, str]]:
    out = []
    for line in master.read_text().splitlines():
        if ";" in line:
            key, _, value = line.partition(";")
            out.append((key.strip(), value.strip()))
    return out


def test_a_441_arm_gets_background_staging_and_reconciled_names(tmp_path, monkeypatch):
    calls = _record_java_reps(monkeypatch)
    xeng.ensemble("4.4.1", BALTIC, "baltic", years=1, n=1, spinup=0, tmp=tmp_path)

    (call,) = calls
    names = [v for k, v in _entries(call["master"]) if k.startswith("species.name.sp")]
    assert names and not [n for n in names if "_" in n], (
        f"4.4.1 cannot resolve these staged names (GitHub #138): {[n for n in names if '_' in n]}"
    )
    assert call["overrides"] and "output.cutoff.enabled" in call["overrides"], (
        f"background-staging overrides did not reach Java: {call['overrides']}"
    )


def test_java_results_come_back_under_the_configs_species_names(tmp_path, monkeypatch):
    """Staging strips ``_`` for 4.4.x, so Java reports ``codwest``; Python reports ``cod_west``.

    Unmapped, the two cod stocks were dropped from every Baltic comparison as "missing from
    >=1 present arm" (seen on a real 2-rep run, 2026-11-07).
    """

    def fake_java_rep(ver, master, years, odir, spinup, prefix, overrides=None):
        return {m: {"codwest": 2.0, "herring": 1.0} for m in xeng.METRICS + xeng.OPTIONAL_METRICS}

    monkeypatch.setattr(xeng, "java_rep", fake_java_rep)
    out = xeng.ensemble("4.4.1", BALTIC, "baltic", years=1, n=1, spinup=0, tmp=tmp_path)
    assert set(out["biomass"]) == {"cod_west", "herring"}


def test_a_433_arm_carries_the_step_count_once(tmp_path, monkeypatch):
    calls = _record_java_reps(monkeypatch)
    xeng.ensemble("4.3.3", BOB, "biscay", years=1, n=1, spinup=0, tmp=tmp_path)

    (call,) = calls
    values = [v for k, v in _entries(call["master"]) if k == "species.biomass.nsteps.year"]
    assert values == ["24"], f"expected the 4.3.3 step count exactly once, got {values}"
