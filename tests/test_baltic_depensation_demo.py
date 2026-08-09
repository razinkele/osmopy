"""Tests for the baltic_depensation overlay scaffold (SP1 Unit 4).

The overlay enables the recruitment depensation/Allee gate for cod on top of the baltic
demo. Its operating-point values (s50/theta/larval-M) are PLACEHOLDERS marked TBD from the
placement sweep (Task 8); these tests only assert the scaffold loads, is Python-only
(Java-guarded), and passes strict validation.
"""

from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import demo_info, list_demos, osmose_demo
from osmose.engine.config_validation import validate
from osmose.runner import java_engine_block_reason

DATA = Path(__file__).resolve().parent.parent / "data"
DEP_DIR = DATA / "baltic_depensation"


def _parse_csv(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition(";")
        d[k.strip()] = v.strip()
    return d


def _includes(path: Path) -> dict[str, str]:
    return {k: v for k, v in _parse_csv(path).items() if k.startswith("osmose.configuration.")}


def test_depensation_registered_python_only():
    assert "baltic_depensation" in list_demos()
    info = demo_info("baltic_depensation")
    assert info is not None
    for field in ("title", "region", "species", "resources", "engine", "summary"):
        assert info.get(field), f"DEMO_INFO['baltic_depensation'] missing {field}"
    assert info["engine"] == "Python"


def test_depensation_generates_and_loads(tmp_path):
    out = osmose_demo("baltic_depensation", tmp_path)
    cfg = Path(out["config_file"])
    assert cfg.name == "baltic_depensation_all-parameters.csv" and cfg.exists()
    # Overlay is text-only (no NetCDF duplication).
    assert not any(p.suffix == ".nc" for p in DEP_DIR.iterdir())
    loaded = dict(OsmoseConfigReader().read(str(cfg)))
    assert loaded["reproduction.depensation.gate.enabled"] == "true"
    assert loaded["reproduction.depensation.gate.species.enabled.sp0"] == "true"
    assert float(loaded["reproduction.depensation.gate.s50.sp0"]) > 0.0
    assert float(loaded["reproduction.depensation.gate.theta.sp0"]) >= 1.0
    assert loaded["simulation.time.nyear"] == "15"  # inherited from baltic


def test_depensation_master_includes_parity():
    # Same includes as baltic (gate keys are inline in the master, no new include file), EXCEPT
    # osmose.configuration.oxygen (spec Phase 2a, adopted 2026-08-09): baltic-only, deliberately
    # NOT propagated here — the depensation/Allee gate stacked with the O2->benthos coupling has
    # never been A/B gated, so this demo stays on baltic's pre-oxygen include set.
    baltic_inc = _includes(DATA / "baltic" / "baltic_all-parameters.csv")
    dep_inc = _includes(DEP_DIR / "baltic_depensation_all-parameters.csv")
    assert set(dep_inc) == set(baltic_inc) - {"osmose.configuration.oxygen"}


def test_depensation_blocks_java_engine(tmp_path):
    out = osmose_demo("baltic_depensation", tmp_path)
    loaded = dict(OsmoseConfigReader().read(str(out["config_file"])))
    # Pin jar 4.4.1 so the nbackground/staging path returns None; the ONLY thing that must
    # block Java here is the depensation-gate guard (a Python-only feature).
    reason = java_engine_block_reason(loaded, jar_version="4.4.1")
    assert reason is not None
    assert "depensation" in reason.lower()


def test_depensation_passes_strict_validation(tmp_path):
    out = osmose_demo("baltic_depensation", tmp_path)
    loaded = dict(OsmoseConfigReader().read(str(out["config_file"])))
    assert validate(loaded, "error") == []
