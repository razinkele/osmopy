# NOTE: import only what THIS task's tests use, so each commit stays ruff-clean (no F401).
# Tasks 2 and 3 add their own imports (pytest, osmose_demo, OsmoseConfigReader, validate,
# java_engine_block_reason, ...) to this header when they add the tests that use them.
from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import demo_info, list_demos, osmose_demo
from osmose.engine.config_validation import validate

DATA = Path(__file__).resolve().parent.parent / "data"
BALTIC_A2_DIR = DATA / "baltic_a2"

NDT = 24  # baltic simulation.time.ndtperyear

# Converged ENGINE-SPACE values (a2_on_converged.params) — what the engine must receive.
# CRITICAL UNIT NOTE: OsmoseConfigReader divides every mortality.additional.larva.rate.spN by
# NDT on load (osmose.version >= 4.4.0; osmose/config/reader.py:100-104 via aliases._LARVA_RATE_RE).
# The DE calibrated in this divided/engine space and injected overrides AFTER the reader, so the
# larval CSV must store converged x NDT (exactly like baltic stores 360.0 to yield 15.0). Adult
# mortality.additional.rate.spN and all species.regrowth.rate.* are NOT matched by the regex -> stored verbatim.
CONVERGED_LARVA = {
    "mortality.additional.larva.rate.sp0": 1.8495054614929225,
    "mortality.additional.larva.rate.sp1": 0.6091614461276307,
    "mortality.additional.larva.rate.sp2": 1.7574285062912955,
    "mortality.additional.larva.rate.sp3": 0.3277205467582994,
    "mortality.additional.larva.rate.sp4": 5.024141712395672,
    "mortality.additional.larva.rate.sp5": 1.1869723413415985,
    "mortality.additional.larva.rate.sp6": 0.3791432328547528,
    "mortality.additional.larva.rate.sp7": 0.27314862986759136,
}
CONVERGED_ADULT = {
    "mortality.additional.rate.sp0": "4.288045380663061",
    "mortality.additional.rate.sp1": "0.2636287453341465",
    "mortality.additional.rate.sp2": "0.003071941136699811",
    "mortality.additional.rate.sp3": "0.0045211280482306045",
    "mortality.additional.rate.sp4": "0.005680413608708062",
    "mortality.additional.rate.sp5": "0.855951786667689",
    "mortality.additional.rate.sp6": "0.0036156979635421347",
    "mortality.additional.rate.sp7": "0.19494616193531136",
}
# Exactly what the CSV stores for larval rates (= converged x NDT; = repr(conv*24)). Literal
# strings so the raw-file parse test is an exact string compare.
STORED_LARVA = {
    "mortality.additional.larva.rate.sp0": "44.38813107583014",
    "mortality.additional.larva.rate.sp1": "14.619874707063136",
    "mortality.additional.larva.rate.sp2": "42.17828415099109",
    "mortality.additional.larva.rate.sp3": "7.865293122199186",
    "mortality.additional.larva.rate.sp4": "120.57940109749615",
    "mortality.additional.larva.rate.sp5": "28.487336192198363",
    "mortality.additional.larva.rate.sp6": "9.099437588514068",
    "mortality.additional.larva.rate.sp7": "6.555567116822193",
}
EXPECTED_MORTALITY_RAW = {**STORED_LARVA, **CONVERGED_ADULT}  # what the CSV literally contains
EXPECTED_DEPLETION = {
    "ltl.depletable.enabled": "true",
    "ltl.depletable.floor": "0.05",
    "species.regrowth.rate.sp8": "5.0",
    "species.regrowth.rate.sp9": "5.0",
    "species.regrowth.rate.sp10": "1.0580953986747008",
    "species.regrowth.rate.sp11": "1.0580953986747008",
    "species.regrowth.rate.sp12": "1.0580953986747008",
    "species.regrowth.rate.sp13": "1.0580953986747008",
}


def _parse_csv(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition(";")
        d[k.strip()] = v.strip()
    return d


def test_a2_mortality_deltas_exact():
    got = _parse_csv(BALTIC_A2_DIR / "baltic_a2_param-additional-mortality.csv")
    assert got == EXPECTED_MORTALITY_RAW


def test_a2_depletion_deltas_exact():
    got = _parse_csv(BALTIC_A2_DIR / "baltic_a2_param-depletion.csv")
    assert got == EXPECTED_DEPLETION


def test_a2_registered_python_only():
    assert "baltic_a2" in list_demos()
    info = demo_info("baltic_a2")
    assert info is not None
    for field in ("title", "region", "species", "resources", "engine", "summary"):
        assert info.get(field), f"DEMO_INFO['baltic_a2'] missing {field}"
    assert info["engine"] == "Python"
    assert "a2" in info["title"].lower() or "calibrat" in info["title"].lower()


def test_a2_generates_and_loads(tmp_path):
    out = osmose_demo("baltic_a2", tmp_path)
    cfg = Path(out["config_file"])
    assert cfg.name == "baltic_a2_all-parameters.csv" and cfg.exists()
    # Overlay must NOT duplicate NetCDFs: baltic_a2 dir is text-only.
    assert not any(p.suffix == ".nc" for p in BALTIC_A2_DIR.iterdir())
    # Loads cleanly through the reader (proves basename includes resolve after overlay).
    loaded = dict(OsmoseConfigReader().read(str(cfg)))
    # Depletion keys are STRINGS (never float('true')) and are not migrated -> exact match.
    for key, val in EXPECTED_DEPLETION.items():
        assert loaded[key] == val, f"{key}: {loaded[key]!r} != {val!r}"
    # Larval rates: reader divides by NDT and reformats via .10g -> the ENGINE receives the
    # converged per-cohort value. Compare with tolerance (.10g truncates to ~10 sig figs).
    for key, conv in CONVERGED_LARVA.items():
        assert float(loaded[key]) == pytest.approx(conv, rel=1e-6), key
    # Adult rates: not migrated -> engine gets the verbatim converged value.
    for key, val in CONVERGED_ADULT.items():
        assert float(loaded[key]) == pytest.approx(float(val), rel=1e-9), key
    assert loaded["simulation.time.nyear"] == "15"  # inherited from baltic


def test_a2_passes_strict_validation(tmp_path):
    # The new include key osmose.configuration.a2.depletion must be allowlisted so baltic_a2 is
    # clean under strict validation (validate() returns [] and does not raise on mode "error").
    out = osmose_demo("baltic_a2", tmp_path)
    loaded = dict(OsmoseConfigReader().read(str(out["config_file"])))
    assert validate(loaded, "error") == []


def _includes(path: Path) -> dict[str, str]:
    return {k: v for k, v in _parse_csv(path).items() if k.startswith("osmose.configuration.")}


def test_a2_master_includes_parity(tmp_path):
    baltic_inc = _includes(DATA / "baltic" / "baltic_all-parameters.csv")
    a2_inc = _includes(BALTIC_A2_DIR / "baltic_a2_all-parameters.csv")
    # Same include KEYS plus the one new depletion include.
    assert set(a2_inc) == set(baltic_inc) | {"osmose.configuration.a2.depletion"}
    # Every include TARGET basename exists in the generated config dir.
    out = osmose_demo("baltic_a2", tmp_path)
    cfgdir = Path(out["config_file"]).parent
    for target in a2_inc.values():
        assert (cfgdir / target).exists(), f"include target missing: {target}"
