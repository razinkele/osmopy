# NOTE: import only what THIS task's tests use, so each commit stays ruff-clean (no F401).
# Tasks 2 and 3 add their own imports (pytest, osmose_demo, OsmoseConfigReader, validate,
# java_engine_block_reason, ...) to this header when they add the tests that use them.
from pathlib import Path

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
