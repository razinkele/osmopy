"""CI-safe tests for the F1 hindcast harness helpers (spec §4, decisions 6-7).
The hindcast RUN is not a CI gate; these cover only the pure functions."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "baltic_f_hindcast",
    Path(__file__).resolve().parent.parent / "scripts" / "baltic_f_hindcast.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

YEARS = list(range(1993, 2024))


def test_annualize_per_step_and_per_year():
    per_year = np.arange(50.0)
    assert (m.annualize(per_year, 50) == per_year).all()
    per_step = np.repeat(np.arange(50.0), 24)
    assert (m.annualize(per_step, 50) == np.arange(50.0)).all()


def test_decadal_trend_signs():
    rising_then_falling = (
        [float(y) for y in YEARS[:10]] + [0.0] * 10 + [float(2024 - y) for y in YEARS[20:]]
    )
    signs = m.decadal_trend_signs(rising_then_falling, YEARS)
    assert signs[0] == 1 and signs[2] == -1


def test_skill_verdict_margins():
    # spec decision 7: mean dr >= 0.10 AND mean dr > 2*sd
    assert m.skill_verdict([0.12, 0.11, 0.13, 0.12, 0.12])["passes"] is True
    assert m.skill_verdict([0.009, 0.010, 0.008, 0.011, 0.009])["passes"] is False  # July spike
    assert m.skill_verdict([0.30, -0.10, 0.25, -0.05, 0.15])["passes"] is False  # noisy


def test_zscore_unit_variance():
    z = m.zscore([1.0, 2.0, 3.0, 4.0])
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1.0) < 1e-12
    assert (m.zscore([2.0, 2.0, 2.0]) == 0).all()


def test_observed_herring_z_is_catch_share_weighted(tmp_path):
    """Two synthetic stocks, opposite SSB trends, 3:1 mean catch share ->
    composite must tilt to the big stock's trend (spec decision 6)."""
    import json

    def snap(key, ssb_by_year, catches):
        recs = [
            {"year": str(y), "ssb": str(ssb_by_year(y)), "f": "0.2", "catches": str(catches)}
            for y in YEARS
        ]
        (tmp_path / f"{key}.assessment.json").write_text(json.dumps(recs))

    snap("her.27.25-2932", lambda y: y - 1990, 300.0)  # rising, weight 3
    snap("her.27.28", lambda y: 2030 - y, 100.0)  # falling, weight 1
    snap("her.27.3031", lambda y: 1.0, 0.0)
    snap("her.27.20-24", lambda y: 1.0, 0.0)
    z = m.observed_herring_z(tmp_path, YEARS)
    assert len(z) == len(YEARS)
    assert z[-1] > z[0]  # composite follows the dominant stock upward
