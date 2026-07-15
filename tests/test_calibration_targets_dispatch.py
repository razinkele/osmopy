"""Dispatch generalization for reference_point_type calibration targets."""

from __future__ import annotations

import pytest

from osmose.calibration.losses import STABILITY_TYPES, make_banded_objective, quantity_key
from osmose.calibration.targets import BiomassTarget


def test_quantity_key_maps_types():
    assert quantity_key("biomass") == "_mean"
    assert quantity_key("ssb") == "_mean"
    assert quantity_key("catch") == "_yield_mean"


def test_quantity_key_rejects_unknown():
    with pytest.raises(ValueError, match="reference_point_type"):
        quantity_key("landings")


def test_stability_types_excludes_catch():
    assert "biomass" in STABILITY_TYPES and "ssb" in STABILITY_TYPES
    assert "catch" not in STABILITY_TYPES


def test_biomass_only_backward_compatible():
    # One biomass target per species -> identical to the pre-change behaviour.
    targets = [
        BiomassTarget("cod", 100.0, 50.0, 200.0, weight=1.0, reference_point_type="biomass"),
        BiomassTarget("sprat", 1000.0, 500.0, 2000.0, weight=0.5, reference_point_type="biomass"),
    ]
    obj, _ = make_banded_objective(targets, ["cod", "sprat"])
    stats = {
        "cod_mean": 30.0,
        "cod_cv": 0.1,
        "cod_trend": 0.0,  # below band -> log10(50/30)**2
        "sprat_mean": 1000.0,
        "sprat_cv": 0.0,
        "sprat_trend": 0.0,
    }  # in band -> 0
    import math

    cod_err = 1.0 * math.log10(50.0 / 30.0) ** 2
    expected = cod_err + 0.0 + 0.5 * max(cod_err, 0.0)  # total + w_worst*worst
    # Strict equality: backward-compat parity with the pre-change objective must be
    # EXACT, not approximate (fuzz-verified bit-identical by a prior reviewer).
    assert obj(stats) == expected


def test_catch_target_dispatches_to_yield_and_no_stability():
    # cod has BOTH a biomass and a catch target; catch reads _yield_mean and adds NO CV/trend penalty.
    targets = [
        BiomassTarget("cod", 100.0, 50.0, 200.0, weight=1.0, reference_point_type="biomass"),
        BiomassTarget("cod", 800.0, 400.0, 1600.0, weight=0.5, reference_point_type="catch"),
    ]
    obj, _ = make_banded_objective(targets, ["cod"])
    # biomass in band (100) -> 0; catch below band (200 < 400) -> log10(400/200)**2, weight 0.5.
    # A high CV must NOT add a penalty for the catch target.
    stats = {"cod_mean": 100.0, "cod_cv": 0.9, "cod_trend": 0.9, "cod_yield_mean": 200.0}
    import math

    catch_err = 0.5 * math.log10(400.0 / 200.0) ** 2
    # biomass target: in band -> 0 error, but its CV(0.9)>0.2 & trend(0.9)>0.05 DO add stability.
    stab = 5.0 * 1.0 * (0.9 - 0.2) ** 2 + 5.0 * 1.0 * (0.9 - 0.05) ** 2
    expected = 0.0 + catch_err + stab + 0.5 * max(0.0, catch_err)
    assert obj(stats) == expected


def test_missing_yield_stat_penalizes_catch_target():
    targets = [BiomassTarget("cod", 800.0, 400.0, 1600.0, weight=0.5, reference_point_type="catch")]
    obj, _ = make_banded_objective(targets, ["cod"])
    stats = {"cod_mean": 100.0}  # no cod_yield_mean
    # missing quantity -> 100.0 penalty (matches the existing missing-_mean path), weighted? No:
    # the missing path adds a flat 100.0 (as today) + w_worst*100.0.
    assert obj(stats) == 100.0 + 0.5 * 100.0


def test_unknown_reference_point_type_raises_at_construction():
    # Must fail loud at make_banded_objective(...) time, not on the first
    # objective(stats) call after a wasted simulation.
    targets = [BiomassTarget("cod", 100.0, 50.0, 200.0, reference_point_type="bogus")]
    with pytest.raises(ValueError, match="reference_point_type"):
        make_banded_objective(targets, ["cod"])
