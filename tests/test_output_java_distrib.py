"""Java-rule distribution outputs (osmose/engine/output_distrib.py; Java OutputDistribution,
DistribOutput, WeightedDistribOutput): biomass by TL, mean size by age, mean TL by size."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.output_distrib import DEFAULTS, class_labels, java_classes, java_thresholds


def test_defaults_match_java():
    t = java_thresholds({}, "tl")
    assert len(t) == 51 and t[0] == np.float32(1.0) and t[-1] == np.float32(6.0)
    assert len(java_thresholds({}, "age")) == 25 and len(java_thresholds({}, "size")) == 20
    assert DEFAULTS["size"] == (10.0, 200.0, 10.0)


def test_thresholds_are_float32_min_plus_i_incr():
    t = java_thresholds({}, "tl")
    expect = np.float32(1.0) + np.arange(51, dtype=np.float32) * np.float32(0.1)
    assert t.dtype == np.float32 and t.tobytes() == expect.astype(np.float32).tobytes()


def test_keys_apply_only_when_all_three_are_set_and_min_zero_is_corrected():
    partial = {"output.distrib.bytl.min": "2", "output.distrib.bytl.max": "4"}
    assert java_thresholds(partial, "tl").tobytes() == java_thresholds({}, "tl").tobytes()
    full = partial | {"output.distrib.bytl.incr": "0.5"}
    assert java_thresholds(full, "tl").tolist() == [2.0, 2.5, 3.0, 3.5, 4.0]
    zero = {
        "output.distrib.byage.min": "0",
        "output.distrib.byage.max": "3",
        "output.distrib.byage.incr": "1",
    }
    assert java_thresholds(zero, "age").tolist()[0] == 1.0  # Java: min 0 -> incr


def test_class_is_count_of_thresholds_at_or_below():
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert java_classes(np.array([0.5, 1.0, 1.5, 3.0, 9.0]), t).tolist() == [0, 1, 1, 3, 3]
    assert class_labels(t) == ["0", "1", "2", "3"]


def _out(config_extra=None):
    """A real config with the three families on, plus a tiny state."""
    from pathlib import Path

    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig

    root = Path(__file__).resolve().parent.parent
    raw = OsmoseConfigReader().read(next((root / "data" / "eec_full").glob("*all-parameters*.csv")))
    raw |= {
        "output.biomass.bytl.enabled": "true",
        "output.meansize.byage.enabled": "true",
        "output.meantl.bysize.enabled": "true",
    } | (config_extra or {})
    return EngineConfig.from_dict(raw)


def _state(config, length, weight, abundance, tl, age_years, sp=0):
    from osmose.engine.state import SchoolState

    n = len(length)
    s = SchoolState.create(n, species_id=np.full(n, sp, dtype=np.int32))
    return s.replace(
        length=np.asarray(length, float),
        weight=np.asarray(weight, float),
        abundance=np.asarray(abundance, float),
        biomass=np.asarray(abundance, float) * np.asarray(weight, float),
        trophic_level=np.asarray(tl, float),
        age_dt=(np.asarray(age_years) * config.n_dt_per_year).astype(np.int32),
    )


def test_window_mean_is_ratio_of_sums_not_mean_of_ratios():
    """Java WeightedDistribOutput sums numerator and denominator over the record window."""
    from osmose.engine.output import _build_java_distribution_dataframes
    from osmose.engine.simulate import _average_step_outputs, _collect_outputs

    c = _out()
    # same size class, two steps: step 1 has 1 fish of TL 2, step 2 has 99 fish of TL 4
    s1 = _state(c, [25.0], [1.0], [1.0], [2.0], [3.0])
    s2 = _state(c, [25.0], [1.0], [99.0], [4.0], [3.0])
    rec = _average_step_outputs([_collect_outputs(s1, c, 0), _collect_outputs(s2, c, 1)], 2, 1)
    frames = _build_java_distribution_dataframes([rec], c, np.array([0.0]))
    sp0 = c.species_names[0]
    row = frames[f"meanTLBySize_{sp0}"].iloc[0, 1:].to_numpy(dtype=float)
    finite = row[np.isfinite(row)]
    assert len(finite) == 1
    assert finite[0] == pytest.approx((1 * 2 + 99 * 4) / 100)  # 3.98, not (2 + 4) / 2 = 3
    assert np.isnan(row).sum() == len(row) - 1  # empty classes are NaN, not 0


def test_no_cutoff_is_applied():
    from osmose.engine.simulate import _collect_outputs

    c = _out({"output.cutoff.age.sp0": "5", "output.cutoff.enabled": "true"})
    young = _state(c, [25.0], [1.0], [10.0], [3.0], [0.5])  # below the cutoff age
    out = _collect_outputs(young, c, 0)
    assert out.biomass_by_tl[0].sum() == pytest.approx(10.0)  # still counted (DistribOutput)
    assert out.meansize_byage_den[0].sum() == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("mn", "mx", "incr"),
    [("1", "6", "0"), ("1", "6", "-1"), ("6", "1", "1"), ("1", "nan", "1"), ("1", "1e9", "0.001")],
)
def test_bad_or_huge_bins_are_a_config_error(mn, mx, incr):
    cfg = {
        "output.distrib.bytl.min": mn,
        "output.distrib.bytl.max": mx,
        "output.distrib.bytl.incr": incr,
    }
    with pytest.raises(ValueError, match="output.distrib.bytl"):
        java_thresholds(cfg, "tl")


# ------------------------------------------------------------------ catch-derived (PR-B)

CATCH_ON = {
    "output.yield.biomass.bysize.enabled": "true",
    "output.size.catch.enabled": "true",
    "output.tl.catch.enabled": "true",
    "output.mortality.perspecies.bysize.enabled": "true",
}


def _caught(c, length, weight, abundance, tl, f_dead, age_years=3.0):
    from osmose.engine.state import MortalityCause

    s = _state(c, length, weight, abundance, tl, [age_years] * len(length))
    n_dead = np.zeros_like(s.n_dead)
    n_dead[:, int(MortalityCause.FISHING)] = f_dead
    return s.replace(n_dead=n_dead)


def test_catch_means_are_ratios_of_window_sums():
    """Java WeightedSpeciesOutput: numerator and denominator summed over the window."""
    from osmose.engine.output import _build_catch_dataframes
    from osmose.engine.simulate import _average_step_outputs, _collect_outputs

    c = _out(CATCH_ON)
    s1 = _caught(c, [20.0], [2.0], [100.0], [3.0], [1.0])  # 1 fish caught, L 20, TL 3
    s2 = _caught(c, [40.0], [2.0], [100.0], [4.0], [9.0])  # 9 fish caught, L 40, TL 4
    rec = _average_step_outputs([_collect_outputs(s1, c, 0), _collect_outputs(s2, c, 1)], 2, 1)
    frames = _build_catch_dataframes([rec], c, np.array([0.0]))
    sp0 = c.species_names[0]
    assert frames["meanSizeCatch"][sp0].iloc[0] == pytest.approx((20 * 1 + 40 * 9) / 10)  # 38
    assert frames["meanTLCatch"][sp0].iloc[0] == pytest.approx((3 * 2 + 4 * 18) / 20)  # 3.9
    assert np.isnan(frames["meanSizeCatch"][c.species_names[1]].iloc[0])  # no catch -> NaN


def test_yield_by_size_is_summed_not_averaged_and_counts_dead_and_young_schools():
    from osmose.engine.output_distrib import java_classes
    from osmose.engine.simulate import _average_step_outputs, _collect_outputs

    c = _out(CATCH_ON | {"output.cutoff.age.sp0": "5", "output.cutoff.enabled": "true"})
    # a school wiped out this step (abundance 0) and one younger than the cutoff still count
    s = _caught(c, [25.0, 25.0], [2.0, 2.0], [0.0, 50.0], [3.0, 3.0], [4.0, 1.0], age_years=0.5)
    one = _collect_outputs(s, c, 0)
    rec = _average_step_outputs([one, _collect_outputs(s, c, 1)], 2, 1)
    cls = int(java_classes(np.array([25.0]), c.output_size_thresholds)[0])
    assert one.yield_by_size[0][cls] == pytest.approx((4 + 1) * 2.0)
    assert rec.yield_by_size[0][cls] == pytest.approx(2 * (4 + 1) * 2.0)  # SUM over the window


def test_mortality_by_size_is_the_stage_rule_on_size_classes_summed():
    from osmose.engine.simulate import (
        _average_step_outputs,
        _collect_by_java_size,
        _collect_outputs,
        mortality_rates_from_counts,
    )

    c = _out(CATCH_ON)
    s = _caught(c, [5.0, 25.0], [1.0, 2.0], [90.0, 80.0], [3.0, 3.0], [10.0, 20.0])
    a, d = _collect_by_java_size(s, c)
    want = mortality_rates_from_counts(d.transpose(0, 2, 1), a).transpose(0, 2, 1)
    one = _collect_outputs(s, c, 0)
    assert np.array_equal(one.mortality_rate_by_cause_size, want)
    from osmose.engine.state import MortalityCause

    f = int(MortalityCause.FISHING)
    assert (want[0, f] > 0).sum() == 2  # both schools' classes carry a fishing rate
    rec = _average_step_outputs([one, _collect_outputs(s, c, 1)], 2, 1)
    assert np.array_equal(rec.mortality_rate_by_cause_size, 2 * want)
