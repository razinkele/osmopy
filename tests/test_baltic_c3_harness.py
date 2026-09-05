"""Tests for scripts/baltic_c3_bioen_ab.py (C3 bioen Stage-1, Task 12).

Each gate helper is exercised on both a synthetic PASS and a synthetic VIOLATION --
per the task-12 brief, a gate that only ever sees the passing case is not proven to
fire. `run_c3` itself is NOT invoked by this suite (5 seeds x 3 arms x 50 yr is Task
14's deliverable, not this one's) -- see the module docstring of
scripts/baltic_c3_bioen_ab.py for the manual wiring validation performed instead.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c3ab", ROOT / "scripts" / "baltic_c3_bioen_ab.py")
c3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c3)


def test_arm_config_overlay_only_on_bioen_arms():
    base = {"simulation.time.nyear": "50", "predation.ingestion.rate.max.sp0": "3.5"}
    ov = {
        "module.bioenergetics.enabled": "true",
        "predation.ingestion.rate.max.sp0": "4.7",
        "temperature.filename": "/x.nc",
    }
    assert c3.arm_config(base, "baseline", ov) == base
    b = c3.arm_config(base, "bioen", ov)
    assert b["predation.ingestion.rate.max.sp0"] == "4.7" and "temperature.offset" not in b
    p = c3.arm_config(base, "bioen_plus2C", ov)
    assert p["temperature.offset"] == "2.0"
    with pytest.raises(AssertionError, match="bioen-off"):
        c3.arm_config(
            {**base, "module.bioenergetics.enabled": "false"},
            "baseline",
            ov | {"module.bioenergetics.enabled": "false"},
        )


def test_gate_d_fires_on_engine_parsed_mismatch():
    class EC:  # minimal stand-in with the fields Gate D reads
        n_species = 1
        bioen_tp = np.array([12.5])
        bioen_r = np.array([0.3])
        bioen_zlayer = np.array([1])

    fit_vals = {
        "species.bioen.mobilized.tp.sp0": "12.5",
        "species.maturity.r.sp0": "0.3",
        "species.zlayer.sp0": "1",
    }
    c3.gate_d_structure({"temperature.filename": "/x.nc"}, EC(), fit_vals, expected_zlayer={0: 1})
    bad = dict(fit_vals)
    bad["species.bioen.mobilized.tp.sp0"] = "20.0"
    with pytest.raises(AssertionError, match="tp"):
        c3.gate_d_structure({"temperature.filename": "/x.nc"}, EC(), bad, expected_zlayer={0: 1})
    with pytest.raises(AssertionError, match="temperature.value"):
        c3.gate_d_structure(
            {"temperature.filename": "/x.nc", "temperature.value": "5"},
            EC(),
            fit_vals,
            expected_zlayer={0: 1},
        )


def test_gate_d_frames_layers_fires_on_wrong_shape(tmp_path):
    import xarray as xr

    good = xr.Dataset(
        {"temperature": (("time", "layer", "latitude", "longitude"), np.zeros((24, 2, 3, 3)))}
    )
    good_path = tmp_path / "good.nc"
    good.to_netcdf(good_path)
    c3.gate_d_frames_layers(good_path)  # no raise

    bad = xr.Dataset(
        {"temperature": (("time", "layer", "latitude", "longitude"), np.zeros((12, 2, 3, 3)))}
    )
    bad_path = tmp_path / "bad_frames.nc"
    bad.to_netcdf(bad_path)
    with pytest.raises(AssertionError, match="24"):
        c3.gate_d_frames_layers(bad_path)

    bad2 = xr.Dataset({"temperature": (("time", "latitude", "longitude"), np.zeros((24, 3, 3)))})
    bad2_path = tmp_path / "bad_layers.nc"
    bad2.to_netcdf(bad2_path)
    with pytest.raises(AssertionError, match="4-D"):
        c3.gate_d_frames_layers(bad2_path)


def test_gate_c_plus2_is_exact_in_float64():
    raw32 = np.array([[3.7, 8.25], [np.nan, 12.125]], dtype=np.float32)
    base = raw32.astype(np.float64)
    arm = 1.0 * (raw32.astype(np.float64) + 2.0)
    wet = np.array([[True, True], [False, True]])
    c3.assert_plus2_exact(arm, base, wet)  # engine_arm - engine_base == 2.0 exactly on wet cells
    with pytest.raises(AssertionError):
        c3.assert_plus2_exact(arm + 1e-9, base, wet)


def test_gate_f_direction_is_sign_of_topt_minus_tbar():
    # g_net shifts up under +2 C when T_bar < t_opt, down when T_bar > t_opt
    out = c3.gate_f_direction(
        t_bar={"a": 5.0, "b": 26.0},
        t_opt={"a": 10.0, "b": 15.0},
        g_base={"a": 1.0, "b": 1.0},
        g_plus2={"a": 1.2, "b": 0.9},
    )
    assert out == {"a": True, "b": True}
    with pytest.raises(AssertionError, match="direction"):
        c3.gate_f_direction({"a": 5.0}, {"a": 10.0}, {"a": 1.0}, {"a": 0.8})


def test_length_from_age_bins_uses_the_real_in_memory_2d_output_shape():
    """`OsmoseResults.abundance_by_age()`/`biomass_by_age()` (in-memory mode) return LONG
    form -- columns ``time, species, bin, value``, ``bin`` a string age-bin index -- not
    the wide ``Time, age, <species>`` layout the brief's original stub assumed (confirmed
    against a real 2-yr smoke run, task-12-report.md). This test uses the real shape;
    `length_from_age_bins` groups by bin and averages over whatever time window the
    caller already restricted the frames to (a single final year here, a final-decade
    mean in `run_c3`).
    """
    ab = pd.DataFrame(
        {
            "time": [49.0, 49.0],
            "species": ["cod", "cod"],
            "bin": ["0", "1"],
            "value": [1e6, 1e5],
        }
    )
    bb = pd.DataFrame(
        {
            "time": [49.0, 49.0],
            "species": ["cod", "cod"],
            "bin": ["0", "1"],
            "value": [1e6 * 1e-6 * 0.5, 1e5 * 1e-6 * 400.0],
        }
    )
    length = c3.length_from_age_bins(ab, bb, cf=0.0087, b=3.05, species="cod")
    assert length[1] == pytest.approx((400.0 / 0.0087) ** (1 / 3.05))


def test_length_from_age_bins_drops_zero_abundance_bins():
    ab = pd.DataFrame(
        {"time": [1.0, 1.0], "species": ["x", "x"], "bin": ["0", "1"], "value": [0.0, 10.0]}
    )
    bb = pd.DataFrame(
        {"time": [1.0, 1.0], "species": ["x", "x"], "bin": ["0", "1"], "value": [0.0, 1e-5]}
    )
    length = c3.length_from_age_bins(ab, bb, cf=0.01, b=3.0, species="x")
    assert 0 not in length
    assert 1 in length


class _FakeSpeciesOutputRes:
    """Minimal stand-in for `OsmoseResults` exposing only `_read_species_output`, the one
    method `_final_window_mean` calls."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _read_species_output(self, output_type: str, name: str) -> pd.DataFrame:
        return self._df


def test_final_window_mean_reads_the_real_bioen_species_output_contract():
    """`_final_window_mean` (used by the realized-ration and realized-ingestion REPORTED
    sections) consumes `_read_species_output`'s real in-memory contract: capital `Time`, and
    the value column named after the output type itself, plus a `species` column -- NOT the
    lowercase `time`/`value` long-form contract `length_from_age_bins` reads from
    `abundance_by_age()`/`biomass_by_age()` (see the tests above). Commit 8a574ab fixed a
    `KeyError: 'time'` caused by assuming the wrong contract here, but nothing pinned it --
    this test does (task-12-review.md finding 2).
    """
    df = pd.DataFrame(
        {
            "Time": [38.0, 39.0, 40.0],
            "meanEnetFaced": [100.0, 200.0, 300.0],
            "species": ["cod_west", "cod_west", "cod_west"],
        }
    )
    res = _FakeSpeciesOutputRes(df)

    ten_year_window = c3._final_window_mean(res, "meanEnetFaced", "cod_west", window_years=10.0)
    assert ten_year_window == pytest.approx((100.0 + 200.0 + 300.0) / 3.0)

    one_year_window = c3._final_window_mean(res, "meanEnetFaced", "cod_west", window_years=1.0)
    assert one_year_window == pytest.approx(300.0)  # only Time=40.0 survives Time > 39.0


def test_final_window_mean_raises_on_the_wrong_column_contract():
    """Discrimination check for the test above: a frame in the OTHER in-memory family's
    lowercase `time`/`value` shape must not be silently accepted -- `_final_window_mean`
    looks columns up by name (`df["Time"]`, `df[output_type]`), so the wrong contract must
    raise, not quietly compute a number over the wrong column."""
    wrong_shape = pd.DataFrame(
        {
            "time": [38.0, 39.0, 40.0],
            "value": [100.0, 200.0, 300.0],
            "species": ["cod_west", "cod_west", "cod_west"],
        }
    )
    with pytest.raises(KeyError):
        c3._final_window_mean(_FakeSpeciesOutputRes(wrong_shape), "meanEnetFaced", "cod_west")


def _healthy_decision_rule_inputs():
    """final_decade/ration stubs where every assessed stock is healthy: bioen mean equals
    its certified mean (criterion iii dead center), bioen mean equals baseline mean
    (criterion i can't fire), e_over_g comfortably above 0.6 (criterion ii can't fire)."""
    final_decade = {"baseline": {}, "bioen": {}}
    ration = {}
    for name in c3.ASSESSED_STOCKS:
        cert = c3.CERTIFIED_MEANS[name]
        final_decade["baseline"][name] = {"mean": cert}
        final_decade["bioen"][name] = {"mean": cert}
        ration[name] = {"e_over_g": 0.75}
    return final_decade, ration


def test_evaluate_decision_rule_passes_clean_when_everything_is_healthy():
    final_decade, ration = _healthy_decision_rule_inputs()
    out = c3.evaluate_decision_rule(final_decade, ration)
    assert out["failed"] == []
    assert out["undetermined"] == []
    assert out["verdict"] == "STAGE 2: WARRANTED"


def test_evaluate_decision_rule_marks_nan_e_over_g_undetermined_not_pass():
    """task-12-review.md finding 1: g_hat == 0 makes `e_over_g` NaN. Before the fix,
    `nan < 0.6` is `False`, so criterion (ii) silently read as satisfied and the verdict
    printed "STAGE 2: WARRANTED" over a species whose ration never computed. The fix must
    read this as undetermined -- distinct from both pass and fail."""
    final_decade, ration = _healthy_decision_rule_inputs()
    ration["cod_west"]["e_over_g"] = float("nan")

    out = c3.evaluate_decision_rule(final_decade, ration)

    assert out["criteria"]["cod_west"]["ii_ebar_ghat"] == "undetermined"
    assert not any("cod_west" in f for f in out["failed"])
    assert any("cod_west" in u and "(ii)" in u for u in out["undetermined"])
    assert out["verdict"].startswith("UNDETERMINED")
    assert "STAGE 2: WARRANTED" not in out["verdict"]


def test_evaluate_decision_rule_does_not_collapse_undetermined_into_failure():
    """A real failure (herring's bioen mean collapses to 1% of baseline) and an unrelated
    NaN (cod_west's e_over_g) must both surface distinctly -- neither may silently absorb
    the other into a single bucket."""
    final_decade, ration = _healthy_decision_rule_inputs()
    final_decade["bioen"]["herring"]["mean"] = final_decade["baseline"]["herring"]["mean"] * 0.01
    ration["cod_west"]["e_over_g"] = float("nan")

    out = c3.evaluate_decision_rule(final_decade, ration)

    assert any("herring" in f for f in out["failed"])
    assert any("cod_west" in u for u in out["undetermined"])
    assert "CLOSE BY CHARACTERIZATION" in out["verdict"]
    assert "undetermined:" in out["verdict"]
