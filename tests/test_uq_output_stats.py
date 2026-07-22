"""Tests for UQ target-keying and per-species output-stat extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.keying import target_to_output_key
from osmose.calibration.uq.output_stats import compute_uq_stats


class _StubResults:
    """Duck-typed OsmoseResults exposing wide biomass/ssb/yield frames.

    Pass ``None`` for a frame to simulate an output that is not enabled: the
    getter raises, mirroring strict-mode ``OsmoseResults``.
    """

    def __init__(self, biomass=None, ssb=None, yield_biomass=None):
        self._biomass = biomass
        self._ssb = ssb
        self._yield = yield_biomass

    def biomass(self):
        if self._biomass is None:
            raise FileNotFoundError("no biomass output")
        return self._biomass

    def ssb(self):
        if self._ssb is None:
            raise FileNotFoundError("no SSB output")
        return self._ssb

    def yield_biomass(self):
        if self._yield is None:
            raise FileNotFoundError("no yield output")
        return self._yield


def _wide(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"Time": np.arange(n), **cols})


def _target(species: str, rpt: str) -> BiomassTarget:
    return BiomassTarget(
        species=species, target=1.0, lower=0.5, upper=2.0, reference_point_type=rpt
    )


def test_keying_distinct_biomass_ssb_yield():
    assert target_to_output_key(_target("cod", "biomass")) == "cod_biomass_mean"
    assert target_to_output_key(_target("cod", "ssb")) == "cod_ssb_mean"
    assert target_to_output_key(_target("cod", "catch")) == "cod_yield_mean"
    # biomass and ssb must NOT collide (they do in losses.quantity_key).
    assert target_to_output_key(_target("cod", "biomass")) != target_to_output_key(
        _target("cod", "ssb")
    )


def test_keying_unknown_reference_point_type_raises():
    with pytest.raises(ValueError, match="unknown reference_point_type"):
        target_to_output_key(_target("cod", "wat"))


def test_output_stats_all_frames_present():
    n = 20
    bio = _wide(cod=np.full(n, 100.0), herring=np.full(n, 50.0))
    ssb = _wide(cod=np.full(n, 60.0))
    yld = _wide(cod=np.full(n, 10.0))
    results = _StubResults(biomass=bio, ssb=ssb, yield_biomass=yld)
    stats = compute_uq_stats(results, ["cod", "herring"], n_eval_years=10)
    assert stats["cod_biomass_mean"] == pytest.approx(100.0)
    assert stats["herring_biomass_mean"] == pytest.approx(50.0)
    assert stats["cod_ssb_mean"] == pytest.approx(60.0)
    assert stats["cod_yield_mean"] == pytest.approx(10.0)
    # herring has no ssb/yield column -> those keys are absent.
    assert "herring_ssb_mean" not in stats
    assert "herring_yield_mean" not in stats


def test_output_stats_missing_ssb_and_yield_frames_skipped():
    n = 20
    bio = _wide(cod=np.full(n, 100.0))
    results = _StubResults(biomass=bio, ssb=None, yield_biomass=None)
    stats = compute_uq_stats(results, ["cod"])
    assert stats == {"cod_biomass_mean": pytest.approx(100.0)}


def test_output_stats_trailing_window_ignores_early_years():
    vals = np.concatenate([np.zeros(5), np.full(10, 7.0)])  # 15 years
    results = _StubResults(biomass=_wide(cod=vals))
    stats = compute_uq_stats(results, ["cod"], n_eval_years=10)
    assert stats["cod_biomass_mean"] == pytest.approx(7.0)


def test_output_stats_shorter_than_window_uses_all_years():
    results = _StubResults(biomass=_wide(cod=np.full(4, 3.0)))
    stats = compute_uq_stats(results, ["cod"], n_eval_years=10)
    assert stats["cod_biomass_mean"] == pytest.approx(3.0)


def test_output_stats_roundtrip_with_keying():
    n = 12
    results = _StubResults(biomass=_wide(cod=np.full(n, 3.0)), ssb=_wide(cod=np.full(n, 2.0)))
    stats = compute_uq_stats(results, ["cod"])
    for rpt in ("biomass", "ssb"):
        assert target_to_output_key(_target("cod", rpt)) in stats
