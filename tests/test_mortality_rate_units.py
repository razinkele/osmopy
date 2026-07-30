"""`mortalityRate-*.csv` must contain RATES, not counts of dead individuals (GitHub #140).

`_collect_mortality_by_cause` accumulates `state.n_dead`, i.e. numbers of individuals. Writing that
under a "Mortality rates per time step" header put values ~1e13 in a file whose Java counterpart
holds values ~1e-4..1e0, and made cross-engine mortality comparison impossible.

Convention pinned here (Java's, per its own header "To get annual mortality rates, sum the mortality
rates within one year"): instantaneous rates that are ADDITIVE across causes, so

    Z          = -ln(N_end / N_start)          total instantaneous mortality for the step
    m_cause    = (D_cause / D_total) * Z       Baranov-style apportionment by share of deaths

with N_start = N_end + D_total.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.output import _build_mortality_dataframes
from osmose.engine.state import MortalityCause


class _Cfg:
    """Minimal stand-in for EngineConfig — only the fields the builder touches."""

    def __init__(self, names: list[str], n_dt: int = 12):
        self.species_names = names
        self.n_dt_per_year = n_dt
        self.n_species = len(names)


class _Out:
    def __init__(self, step: int, abundance: np.ndarray, mortality: np.ndarray):
        self.step = step
        self.abundance = abundance
        self.mortality_by_cause = mortality


def _one_step(n_end: float, deaths: dict[MortalityCause, float]):
    m = np.zeros((1, len(MortalityCause)), dtype=float)
    for cause, d in deaths.items():
        m[0, int(cause)] = d
    return _Out(0, np.array([n_end], dtype=float), m)


def test_rates_are_instantaneous_and_additive():
    n_end, d_pred, d_add = 600.0, 300.0, 100.0
    out = _one_step(n_end, {MortalityCause.PREDATION: d_pred, MortalityCause.ADDITIONAL: d_add})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]

    n_start = n_end + d_pred + d_add
    z = -np.log(n_end / n_start)
    assert df["Predation"].iloc[0] == pytest.approx((d_pred / (d_pred + d_add)) * z)
    assert df["Additional"].iloc[0] == pytest.approx((d_add / (d_pred + d_add)) * z)

    causes = [c.name.capitalize() for c in MortalityCause]
    assert df[causes].iloc[0].sum() == pytest.approx(z), "per-cause rates must sum to total Z"


def test_no_deaths_gives_zero_rates():
    out = _one_step(1000.0, {})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    causes = [c.name.capitalize() for c in MortalityCause]
    assert df[causes].iloc[0].sum() == pytest.approx(0.0)


def test_empty_species_gives_zero_not_nan():
    """Nothing alive and nothing dead must not produce NaN/inf from a 0/0 denominator."""
    out = _one_step(0.0, {})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    causes = [c.name.capitalize() for c in MortalityCause]
    vals = df[causes].iloc[0].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals)) and np.all(vals == 0.0)


def test_complete_removal_reports_infinite_rate_not_a_finite_one():
    """A species wiped out within one step has a genuinely infinite instantaneous rate.

    Pinned deliberately: clamping it would let a total collapse read as an ordinary rate. Causes
    with no deaths must still be exactly 0 (guarding the 0 * inf -> nan trap).
    """
    out = _one_step(0.0, {MortalityCause.PREDATION: 500.0})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    assert np.isinf(df["Predation"].iloc[0])
    assert df["Starvation"].iloc[0] == 0.0
    assert not np.isnan(df["Starvation"].iloc[0])


def test_rates_are_plausible_magnitudes_not_counts():
    """The regression that motivated #140: counts land at 1e9-1e13, rates do not."""
    out = _one_step(1.0e11, {MortalityCause.ADDITIONAL: 5.0e12, MortalityCause.PREDATION: 5.0e9})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    causes = [c.name.capitalize() for c in MortalityCause]
    vals = df[causes].iloc[0].to_numpy(dtype=float)
    assert np.all(vals < 100.0), f"values look like counts, not rates: {dict(zip(causes, vals))}"
    assert vals.sum() > 0.0
