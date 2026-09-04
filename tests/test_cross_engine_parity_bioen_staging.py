"""Task 9 (C3 Gate B) unit tests for the Java-4.3.3 bioen key staging, the non-degeneracy
precondition helper, and the per-metric "vacuous pass" hardening (R39 fix round) added to
scripts/cross_engine_parity_440.py.

Loaded by path (the script lives under scripts/, not a package) rather than imported normally.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "xeng", ROOT / "scripts" / "cross_engine_parity_440.py"
)
xeng = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(xeng)


def test_inject_java_bioen_keys_appends_bioen_imax_for_every_predator(tmp_path):
    master = tmp_path / "osm_all-parameters.csv"
    master.write_text("predation.ingestion.rate.max.sp0 ; 3.5\nspecies.type.sp15 ; background\n")
    raw = {
        "module.bioenergetics.enabled": "true",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp15": "2.5",
        "species.type.sp15": "background",
        "predation.larval.ingestion.rate.increase.ratio.sp0": "1.0",
        "predation.c.bioen.sp0": "0.0",
        "simulation.nspecies": "1",
    }
    n = xeng.inject_java_bioen_keys(master, raw)
    text = master.read_text()
    assert "predation.ingestion.rate.max.bioen.sp0 ; 3.5" in text
    assert "predation.ingestion.rate.max.bioen.sp15 ; 2.5" in text
    assert (
        "predation.coef.ingestion.rate.max.larvae.bioen.sp15 ; 1.0" in text
        and "predation.c.bioen.sp15 ; 0.0" in text
    )
    assert (
        "predation.ingestion.rate.max.sp0 ; 3.5" in text
    )  # legacy standard key kept (Java reads both)
    assert n >= 4


def test_inject_is_noop_without_bioen(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("a ; 1\n")
    assert xeng.inject_java_bioen_keys(master, {"module.bioenergetics.enabled": "false"}) == 0
    assert master.read_text() == "a ; 1\n"


def test_inject_raises_on_missing_imax_for_a_predator(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("x ; 1\n")
    raw = {"module.bioenergetics.enabled": "true", "simulation.nspecies": "2"}
    import pytest

    with pytest.raises(KeyError):
        xeng.inject_java_bioen_keys(master, raw)


def test_inject_resource_nsteps_year_adds_global_fallback_for_file_forced_resource(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("species.file.sp8 ; some.nc\n")
    raw = {"species.file.sp8": "some.nc", "simulation.time.ndtperyear": "24"}
    n = xeng.inject_java_resource_nsteps_year(master, raw)
    assert n == 1
    assert "species.biomass.nsteps.year ; 24" in master.read_text()


def test_inject_resource_nsteps_year_is_noop_when_already_present(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("x ; 1\n")
    raw = {
        "species.file.sp8": "some.nc",
        "simulation.time.ndtperyear": "24",
        "species.biomass.nsteps.year": "24",
    }
    assert xeng.inject_java_resource_nsteps_year(master, raw) == 0
    assert master.read_text() == "x ; 1\n"


def test_inject_resource_nsteps_year_is_noop_without_file_forced_resources(tmp_path):
    master = tmp_path / "m.csv"
    master.write_text("x ; 1\n")
    raw = {"simulation.time.ndtperyear": "24"}
    assert xeng.inject_java_resource_nsteps_year(master, raw) == 0
    assert master.read_text() == "x ; 1\n"


def test_nondegenerate_flags_species_collapsed_in_too_many_reps():
    import numpy as np

    ens = {"biomass": {"A": np.array([500.0] * 16), "B": np.array([500.0] * 13 + [0.5] * 3)}}
    nd = xeng.nondegenerate(ens, "biomass", n=16, floor=1.0, frac=0.9)
    assert nd == {"A": True, "B": False}


def test_nondegenerate_flags_short_array_as_missing_reps():
    import numpy as np

    # A species array shorter than n (some reps never reported it) fails on the size==n check
    # alone -- no NaN or collapsed value involved, isolating the size check from NaN-handling.
    ens = {"biomass": {"A": np.array([500.0, 500.0, 500.0])}}
    nd = xeng.nondegenerate(ens, "biomass", n=4, floor=1.0, frac=0.9)
    assert nd == {"A": False}


def test_nondegenerate_excludes_nan_reps_from_the_ok_fraction():
    import numpy as np

    # Size-matched (n=4, size=4) so the size check alone can't force the result: 1 NaN among 4
    # reps must count as "not ok" AND stay in the denominator (3/4 = 0.75 < frac=0.9) even though
    # every finite value is comfortably non-collapsed. A denominator computed only over finite
    # entries (nan excluded from both numerator and denominator) would wrongly give 3/3 = 1.0 and
    # pass; this pins the correct (in-denominator) behavior.
    ens = {"biomass": {"A": np.array([500.0, np.nan, 500.0, 500.0])}}
    nd = xeng.nondegenerate(ens, "biomass", n=4, floor=1.0, frac=0.9)
    assert nd == {"A": False}


# --- R39: per-metric "vacuous pass" hardening -------------------------------------------------
# main() previously computed, inline, `sp_all = [s for s in py[m] if all(s in ens[v][m] for v in
# present)]` and silently printed nothing for a metric where sp_all came back empty -- no rows,
# no warning, and nothing added to overall_fail, so a metric that was never actually compared
# (e.g. one engine's CSV for just that metric came back empty) still produced a clean GATE: PASS.
# `comparable_species` + `gate_verdict` (below main()'s per-metric loop) close that gap: an empty
# `comparable_species` result now feeds `uncompared_metrics`, which `gate_verdict` fails on. This
# is a DIFFERENT signal from `tost()`'s `se == 0` short-circuit (KS=nan on a real, non-empty
# comparison with zero measured variance -- see test_tost_se_zero_is_genuine_agreement_not_a_gap
# below): that species/metric pair still reaches `comparable_species` and appears in `sp_all`, it
# just gets an uninformative-but-real `d`/`eq`.


def test_comparable_species_empty_when_one_present_arm_never_reported_the_metric():
    # RED: the "vacuous pass" case this hardens. `4.4.1` has zero entries for `yield` (e.g. its
    # CSV came back empty for that one metric) while the python arm has real species -- before
    # the fix, callers silently iterated zero rows here and nothing failed.
    py_metric = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    ens = {"4.4.1": {"yield": {}}}
    assert xeng.comparable_species(py_metric, ens, ["4.4.1"], "yield") == []


def test_comparable_species_nonempty_when_every_present_arm_has_the_species():
    # GREEN: the real Gate B shape -- every present arm reports every species for this metric.
    py_metric = {"A": [1.0], "B": [2.0]}
    ens = {
        "4.3.3": {"biomass": {"A": [1.1], "B": [2.1]}},
        "4.4.1": {"biomass": {"A": [1.2], "B": [2.2]}},
    }
    assert xeng.comparable_species(py_metric, ens, ["4.3.3", "4.4.1"], "biomass") == ["A", "B"]


def test_comparable_species_drops_only_the_species_missing_from_one_arm():
    py_metric = {"A": [1.0], "B": [2.0]}
    ens = {"4.3.3": {"biomass": {"A": [1.1]}}}  # B never reported by this arm
    assert xeng.comparable_species(py_metric, ens, ["4.3.3"], "biomass") == ["A"]


def test_gate_verdict_fails_on_an_uncompared_metric_even_with_no_other_failures():
    # The R39 fix's core assertion: a metric with zero comparable species must not be a silent
    # PASS, even when every other metric and every other check is clean.
    verdict = xeng.gate_verdict([], ["yield"], [], [])
    assert verdict.startswith("FAIL")
    assert "yield" in verdict


def test_gate_verdict_passes_when_every_metric_was_actually_compared():
    # Proves the hardening doesn't regress a genuinely clean run: no dropped arms, no uncompared
    # metrics, no degenerate species, no TOST failures -- exactly Gate B's real committed shape.
    assert xeng.gate_verdict([], [], [], []) == "PASS"


def test_gate_verdict_priority_empty_arm_beats_uncompared_metric():
    verdict = xeng.gate_verdict(["4.4.1"], ["yield"], [], [])
    assert verdict.startswith("FAIL (dropped arm")


def test_tost_se_zero_is_genuine_agreement_not_a_gap():
    # Contrast case for the R39 hardening: tost()'s se==0 short-circuit (both engines' reps have
    # literally zero measured variance -- the real mean_weight/Hake shape in the committed Gate B
    # log) is a REAL comparison on REAL data, not an absent one. It must still produce a species
    # row (i.e. reach comparable_species successfully) with a valid eq/d, just with ci90=0.0 and
    # KS=nan (ks_2samp is never invoked on this branch).
    import numpy as np

    py_vals = np.array([500.0] * 16)
    jv_vals = np.array([500.0] * 16)
    d, ci, p, eq, ks_p, vr = xeng.tost(py_vals, jv_vals, delta=xeng.np.log10(1.5))
    assert bool(eq) is True
    assert ci == 0.0
    assert np.isnan(ks_p)
    assert np.isnan(vr)
    # And this species is NOT dropped by the R39 hardening -- it has real values in both arms.
    py_metric = {"A": py_vals}
    ens = {"4.3.3": {"mean_weight": {"A": jv_vals}}}
    assert xeng.comparable_species(py_metric, ens, ["4.3.3"], "mean_weight") == ["A"]
