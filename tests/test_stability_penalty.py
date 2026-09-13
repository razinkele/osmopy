"""Unit tests for the SP-A stability_penalty (persistence + envelope + trend + variability)."""

import numpy as np
import pandas as pd
import pytest

from osmose.calibration.stability import stability_penalty


class _T:
    """Minimal BiomassTarget stand-in (species, lower, upper, weight)."""

    def __init__(self, species, lower, upper, weight):
        self.species = species
        self.lower = lower
        self.upper = upper
        self.weight = weight
        self.target = (lower + upper) / 2


def _wide(series: dict, n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {"Time": np.arange(n), **{k: np.asarray(v, float) for k, v in series.items()}}
    )


TGT = [_T("cod", 60000, 250000, 1.0)]


def test_flat_in_envelope_is_zero():
    bio = _wide({"cod": np.full(50, 120000.0)})
    assert stability_penalty(bio, TGT) == pytest.approx(0.0, abs=1e-6)


def test_collapse_is_heavily_penalised():
    bio = _wide({"cod": np.linspace(120000, 1.0, 50)})  # collapses below the floor
    assert stability_penalty(bio, TGT) > 1.0


def test_sub_collapse_decline_tracks_trend():
    # both stay ABOVE the persistence floor (0.1*lo=6000) -> trend/envelope, not persistence, drive it
    gentle = _wide({"cod": np.linspace(120000, 80000, 50)})
    steep = _wide({"cod": np.linspace(120000, 12000, 50)})
    assert stability_penalty(steep, TGT) > stability_penalty(gentle, TGT)


def test_persistence_floor_isolated():
    # held just above the floor vs dipping below it -> the persistence term is what differs
    lo = 60000
    alive = _wide({"cod": np.full(50, 0.2 * lo)})  # below envelope but above the 0.1*lo floor
    extinct = _wide({"cod": np.concatenate([np.full(25, 0.2 * lo), np.full(25, 0.05 * lo)])})
    assert stability_penalty(extinct, TGT) > stability_penalty(alive, TGT)


def test_explosion_is_penalised():
    bio = _wide({"cod": np.linspace(120000, 1e7, 50)})
    assert stability_penalty(bio, TGT) > 1.0


def test_boombust_stickleback_not_punished_for_variance():
    tgt = [_T("stickleback", 50000, 500000, 0.2)]
    osc = _wide({"stickleback": 200000 + 150000 * np.sin(np.arange(50))})  # in-envelope oscillation
    assert stability_penalty(osc, tgt) < 0.5


def test_weight_scales_penalty():
    hi = [_T("cod", 60000, 250000, 1.0)]
    lo = [_T("cod", 60000, 250000, 0.2)]
    bio = _wide({"cod": np.linspace(120000, 1.0, 50)})
    assert stability_penalty(bio, hi) > stability_penalty(bio, lo)
