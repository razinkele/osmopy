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
import pandas as pd
import pytest

from osmose.engine.output import _build_mortality_dataframes
from osmose.engine.simulate import mortality_rates_from_counts
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


class _StagedOut:
    """StepOutput stand-in carrying the (cause, stage) arrays."""

    def __init__(self, step, abundance, mortality, abundance_by_stage, mortality_by_cause_stage):
        self.step = step
        self.abundance = abundance
        self.mortality_by_cause = mortality
        self.abundance_by_stage = abundance_by_stage
        self.mortality_by_cause_stage = mortality_by_cause_stage
        self.mortality_rate_by_cause_stage = None


def test_stage_split_emits_cause_stage_multiindex_matching_java():
    """Java splits every cause by Eggs/Juvenil/Adult; the Python output must too (#140).

    Rates are per stage: each stage gets its own Z from its own survivors and deaths, because a
    stage-level rate apportioned from a whole-population Z would be meaningless.
    """
    n_causes = len(MortalityCause)
    # stages: 0=Eggs, 1=Juvenil, 2=Adult
    abund_stage = np.array([[1000.0, 400.0, 200.0]])  # (1 species, 3 stages) survivors
    mort_stage = np.zeros((1, n_causes, 3))
    mort_stage[0, int(MortalityCause.PREDATION), 1] = 100.0  # juveniles predated
    mort_stage[0, int(MortalityCause.ADDITIONAL), 0] = 500.0  # eggs, additional
    flat = mort_stage.sum(axis=2)
    out = _StagedOut(0, abund_stage.sum(axis=1), flat, abund_stage, mort_stage)
    # Mirror simulate.py: rates are formed per step from that step's counts and survivors.
    out.mortality_rate_by_cause_stage = mortality_rates_from_counts(
        mort_stage.transpose(0, 2, 1), abund_stage
    ).transpose(0, 2, 1)

    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]

    assert isinstance(df.columns, pd.MultiIndex), "columns must be a (cause, stage) MultiIndex"
    assert ("Predation", "Juvenil") in df.columns
    assert ("Additional", "Eggs") in df.columns
    for stage in ("Eggs", "Juvenil", "Adult"):
        assert ("Predation", stage) in df.columns

    # juvenile predation: only cause for that stage -> rate == that stage's own Z
    z_juv = -np.log(400.0 / 500.0)
    assert df[("Predation", "Juvenil")].iloc[0] == pytest.approx(z_juv)
    # eggs saw no predation
    assert df[("Predation", "Eggs")].iloc[0] == pytest.approx(0.0)
    # egg additional mortality uses the EGG denominator, not the whole population
    z_egg = -np.log(1000.0 / 1500.0)
    assert df[("Additional", "Eggs")].iloc[0] == pytest.approx(z_egg)
    # adults lost nothing
    assert df[("Predation", "Adult")].iloc[0] == pytest.approx(0.0)


def test_flat_layout_retained_when_stage_arrays_absent():
    """Older StepOutputs without the stage arrays must still produce the flat frame."""
    out = _one_step(600.0, {MortalityCause.PREDATION: 400.0})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    assert not isinstance(df.columns, pd.MultiIndex)
    assert df["Predation"].iloc[0] == pytest.approx(-np.log(600.0 / 1000.0))


class _RateOut:
    """StepOutput stand-in carrying PRE-COMPUTED per-step rates."""

    def __init__(self, step, rate_by_cause_stage):
        self.step = step
        self.abundance = None
        self.mortality_by_cause = None
        self.abundance_by_stage = None
        self.mortality_by_cause_stage = None
        self.mortality_rate_by_cause_stage = rate_by_cause_stage


def test_precomputed_rates_are_used_verbatim():
    """Java sums per-step rates over the saving interval, so the accumulated value is already the
    rate — output must NOT re-derive it from window-aggregated counts.

    Verified empirically: Java's annual row equals the sum of its 24 per-step rows exactly for the
    deterministic causes (ratio 1.000). Re-deriving a single rate from mean abundance and summed
    deaths has no fixed relationship to that sum, which is what made Python's values non-comparable.
    """
    n_causes = len(MortalityCause)
    rates = np.zeros((1, n_causes, 3))
    rates[0, int(MortalityCause.PREDATION), 2] = 0.4242  # adult predation, already summed
    out = _RateOut(0, rates)

    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    assert df[("Predation", "Adult")].iloc[0] == pytest.approx(0.4242)
    assert df[("Predation", "Juvenil")].iloc[0] == pytest.approx(0.0)


def test_rates_are_plausible_magnitudes_not_counts():
    """The regression that motivated #140: counts land at 1e9-1e13, rates do not."""
    out = _one_step(1.0e11, {MortalityCause.ADDITIONAL: 5.0e12, MortalityCause.PREDATION: 5.0e9})
    df = _build_mortality_dataframes([out], _Cfg(["sprat"]))["mortalityRate_sprat"]
    causes = [c.name.capitalize() for c in MortalityCause]
    vals = df[causes].iloc[0].to_numpy(dtype=float)
    assert np.all(vals < 100.0), f"values look like counts, not rates: {dict(zip(causes, vals))}"
    assert vals.sum() > 0.0
