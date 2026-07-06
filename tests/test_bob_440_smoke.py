# tests/test_bob_440_smoke.py
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from ui.pages.run import write_temp_config

ROOT = Path(__file__).resolve().parents[1]
BOB = ROOT / "data" / "examples" / "osm_all-parameters.csv"


@pytest.mark.skipif(not BOB.exists(), reason="no BoB config")
def test_bob_runs_on_python_engine():
    raw = dict(OsmoseConfigReader().read(str(BOB)))
    raw["simulation.time.nyear"] = "3"  # pin: do NOT inherit nyear;50
    res = PythonEngine().run_in_memory(raw, seed=42)
    bio = res.biomass()
    assert bio is not None and len(bio) > 0
    vals = bio[[c for c in bio.columns if c not in ("Time", "species")]].to_numpy(dtype=float)
    assert np.isfinite(vals).any() and np.nansum(vals) > 0


JAR_441 = ROOT / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar"
_java = shutil.which("java") is not None and JAR_441.exists()


@pytest.mark.skipif(not (_java and BOB.exists()), reason="Java/jar/config unavailable")
def test_bob_runs_on_441_jar(tmp_path):
    raw = dict(OsmoseConfigReader().read(str(BOB)))
    stage = tmp_path / "stage"
    write_temp_config(raw, stage, source_dir=BOB.parent, target_version="4.4.1")
    master = stage / "osm_all-parameters.csv"
    odir = tmp_path / "out"
    odir.mkdir()
    r = subprocess.run(
        [
            "java",
            "-Xmx2g",
            "-jar",
            str(JAR_441),
            str(master),
            f"-Poutput.dir.path={odir}",
            "-Psimulation.time.nyear=3",
            "-Poutput.start.year=0",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    # A load/parameter error unrelated to resource forcing (e.g. legacy fishing on 4.4.1) => RESCOPE.
    assert r.returncode == 0, f"4.4.1 jar failed on BoB:\n{r.stderr[-2000:]}"
    assert list(odir.glob("*.csv")), "no outputs produced"
