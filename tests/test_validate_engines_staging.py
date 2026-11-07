"""``scripts/validate_engines.py`` must give Java a config it can load, and find both outputs.

Until 2026-11-07 the script could not produce a comparison on any jar. It handed Java the
bundled Bay of Biscay config raw, which neither jar loads (4.4.1 wants the ``species.biomass.*``
resource keys the writer synthesizes, 4.3.3 wants ``species.biomass.nsteps.year``). And it looked
for the Python result as ``osmose_biomass_Simu0.csv``, while the engine writes
``osm_biomass_Simu0.csv``. The first two tests need no jar and run in CI; the last runs the real
jars and skips without them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts import validate_engines as ve

ROOT = Path(__file__).resolve().parents[1]
JARS = {
    "4.4.1": ROOT / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar",
    "4.3.3": ROOT / "osmose-java" / "osmose_4.3.3-jar-with-dependencies.jar",
}
# What each jar version must find in its staged master (and the raw config lacks).
_REQUIRED = {"4.4.1": "species.biomass.file.sp", "4.3.3": "species.biomass.nsteps.year"}


def _keys(master: Path) -> set[str]:
    return {ln.partition(";")[0].strip() for ln in master.read_text().splitlines() if ";" in ln}


@pytest.mark.parametrize("version", ["4.4.1", "4.3.3"])
def test_run_java_hands_java_a_config_staged_for_the_jar(version, tmp_path, monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ve.subprocess, "run", fake_run)
    ve.run_java(tmp_path / JARS[version].name, tmp_path / "java", 1)

    cmd = seen["cmd"]
    master = Path(cmd[cmd.index("-jar") + 2])
    assert master != ve.EXAMPLES_CONFIG, "Java was handed the raw bundled config"
    assert any(k.startswith(_REQUIRED[version]) for k in _keys(master)), (
        f"staged master for {version} lacks {_REQUIRED[version]}*; that jar exits without it"
    )
    assert "-Psimulation.time.nyear=1" in cmd
    assert "-Poutput.start.year=0" in cmd


def test_compare_biomass_finds_what_both_engines_write(tmp_path):
    """The Python file is whatever the engine really writes; Java's is named by its prefix."""
    py_dir, java_dir = tmp_path / "python", tmp_path / "java"
    ve.run_python(py_dir, 1, 42)
    java_dir.mkdir()
    (written,) = py_dir.glob("*_biomass_Simu0.csv")
    shutil.copy(written, java_dir / "biscay_biomass_Simu0.csv")  # Java's format is the same

    df = ve.compare_biomass(java_dir, py_dir)
    assert len(df) == 8, f"expected the 8 BoB species, got {list(df.get('Species', []))}"
    assert df["Log10_ratio"].abs().max() == 0.0, "identical files must compare as identical"


def _write_biomass(path: Path, rows: list[dict[str, float]]) -> None:
    """A biomass CSV in the engines' layout: one quoted description line, then the table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write('"Mean biomass (tons), excluding first ages specified in input"\n')
        pd.DataFrame(rows).to_csv(fh, index=False)


def test_compare_biomass_uses_the_final_year_mean_not_the_last_step(tmp_path):
    """0 at the last step in both engines but 100x apart over the year: the gap must show.

    That is BoB Hake on a real 5-yr run (2026-11-07): 0 vs 0 at the final step, which the
    script used to score as agreement, but 247 t vs 41,540 t averaged over year 5.
    """
    times = [(i + 1) / 4 for i in range(8)]  # 2 years, 4 steps a year
    java = [{"Time": t, "Hake": 0.0 if i == 7 else 10.0, "Sole": 5.0} for i, t in enumerate(times)]
    py = [{"Time": t, "Hake": 0.0 if i == 7 else 1000.0, "Sole": 5.0} for i, t in enumerate(times)]
    _write_biomass(tmp_path / "java" / "biscay_biomass_Simu0.csv", java)
    _write_biomass(tmp_path / "python" / "osm_biomass_Simu0.csv", py)

    df = ve.compare_biomass(tmp_path / "java", tmp_path / "python").set_index("Species")
    assert df.loc["Hake", "Log10_ratio"] == pytest.approx(2.0)
    assert not df.loc["Hake", "Within_1_OoM"]
    assert df.loc["Sole", "Log10_ratio"] == pytest.approx(0.0)


_HAVE = {v: shutil.which("java") is not None and j.exists() for v, j in JARS.items()}


@pytest.mark.parametrize("version", ["4.4.1", "4.3.3"])
def test_end_to_end_on_the_real_jar(version, tmp_path):
    if not _HAVE[version]:
        pytest.skip(f"needs `java` and osmose-java/ {version} jar")
    ve.run_java(JARS[version], tmp_path / "java", 1)
    ve.run_python(tmp_path / "python", 1, 42)
    df = ve.compare_biomass(tmp_path / "java", tmp_path / "python")
    assert len(df) == 8, f"expected the 8 BoB species, got {list(df.get('Species', []))}"
    # Final-year means: every species, Hake included, is present in both engines in year 1.
    assert df["Java_biomass"].gt(0).all() and df["Python_biomass"].gt(0).all(), df.to_string()
