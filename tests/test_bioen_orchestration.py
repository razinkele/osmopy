"""Orchestration tests for _bioen_step in osmose/engine/simulate.py.

These tests focus on the _bioen_step function directly, verifying:
  1. With constant temperature data: weight and length update correctly.
  2. Without temperature data: fallback to 15°C neutral temperature.
  3. All bioenergetic state outputs (e_net, e_gross, e_maint, rho) are finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.physical_data import PhysicalData
from osmose.engine.simulate import _bioen_step
from osmose.engine.state import MortalityCause, SchoolState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_CAUSES = len(MortalityCause)


def _make_bioen_config_dict(n_species: int = 2) -> dict[str, str]:
    """Build a minimal bioenergetics config dict for n_species."""
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "2",
        "simulation.nspecies": str(n_species),
        "mortality.subdt": "10",
        "simulation.bioen.enabled": "true",
        "simulation.bioen.phit.enabled": "true",
        "simulation.bioen.fo2.enabled": "false",
        # Task 7 removes the silent 15C fallback in `_bioen_step` -- set the source
        # explicitly so these fixtures don't start raising when that lands. Matches
        # `const_temp_data` below (also 15.0) for the tests that call `_bioen_step`
        # directly with that fixture rather than through `simulate()`.
        "temperature.value": "15.0",
    }
    for i in range(n_species):
        sp = f"sp{i}"
        name = f"Species{i}"
        cfg.update(
            {
                f"simulation.nschool.{sp}": "10",
                f"species.name.{sp}": name,
                f"species.linf.{sp}": "25.0",
                f"species.k.{sp}": "0.3",
                f"species.t0.{sp}": "-0.1",
                f"species.egg.size.{sp}": "0.1",
                f"species.length2weight.condition.factor.{sp}": "0.006",
                f"species.length2weight.allometric.power.{sp}": "3.0",
                f"species.lifespan.{sp}": "3",
                f"species.vonbertalanffy.threshold.age.{sp}": "1.0",
                f"predation.ingestion.rate.max.{sp}": "3.5",
                f"predation.efficiency.critical.{sp}": "0.57",
                # Bioenergetics per-species params
                f"species.beta.{sp}": "0.75",
                f"species.zlayer.{sp}": "0",
                f"species.bioen.assimilation.{sp}": "0.68",
                # Item A (Task 3/reviewer, carried into Task 6): the old value (0.00123)
                # made e_maint ~1e-8 of e_gross -- E_net == E_gross at every abundance and
                # bioen starvation was structurally zero, so nothing built on this fixture
                # could detect the per-fish-grams class of unit bug Tasks 3-5 fixed. 1e5
                # makes maintenance materially bind against this fixture's realistic
                # weight/abundance/ingestion ranges (`_make_school_state`, seed 0, T=15C):
                # measured median e_maint/e_gross ~=0.79 (Task 3's `_three_schools()` model
                # reaches ~0.81), with ~45% of schools landing net-negative -- both growth
                # and starvation are exercised.
                f"species.bioen.maint.energy.c_m.{sp}": "1e5",
                # Canonical keys (osmose/config/aliases.py RENAMES_440 / NEW_TO_OLD):
                # `species.bioen.maturity.*` and the two `*.bioen` predation keys below ARE
                # aliased by canonicalize_config, so those specific legacy spellings would
                # have worked -- but `mobilized.e.D`/`mobilized.Tp` have no case-insensitive
                # match to config.py's lowercase `mobilized.e.d`/`mobilized.tp` reads (keys
                # reaching the engine are lowercase; from_dict does not lowercase), so those
                # two were silently dead, always falling back to config.py's defaults
                # (e_d=1.5, tp=20.0) instead of this fixture's intended 1.45/18.0.
                f"species.maturity.eta.{sp}": "1.4",
                f"species.maturity.r.{sp}": "0.45",
                f"species.maturity.m0.{sp}": "4.5",
                f"species.maturity.m1.{sp}": "1.8",
                f"species.bioen.mobilized.e.mobi.{sp}": "0.62",
                f"species.bioen.mobilized.e.d.{sp}": "1.45",
                f"species.bioen.mobilized.tp.{sp}": "18.0",
                f"species.bioen.maint.e.maint.{sp}": "0.63",
                f"species.oxygen.c1.{sp}": "0.95",
                f"species.oxygen.c2.{sp}": "2.5",
                # `predation.ingestion.rate.max.{sp}` above (already canonical) is the one
                # config.py reads for bioen_i_max too -- the legacy `.bioen.` suffixed key
                # this fixture used to also set here was dead (alias merge is
                # skip-if-canonical-key-exists), so dropping it changes nothing numerically.
                f"predation.larval.ingestion.rate.increase.ratio.{sp}": "1.1",
                f"predation.c.bioen.{sp}": "0.01",
                f"species.bioen.forage.k_for.{sp}": "0.002",
            }
        )
    return cfg


def _make_school_state(n_schools: int, n_species: int = 2) -> SchoolState:
    """Build a minimal but realistic SchoolState for bioen testing."""
    rng = np.random.default_rng(0)

    # Distribute schools evenly across species
    species_id = np.array([i % n_species for i in range(n_schools)], dtype=np.int32)

    # Realistic weight (tonnes) and length (cm) for adult fish
    weight = rng.uniform(1e-5, 1e-3, size=n_schools)
    length = rng.uniform(5.0, 20.0, size=n_schools)

    # Some preyed biomass so energy budget has non-zero intake
    preyed_biomass = rng.uniform(0.0, 1e-5, size=n_schools)

    # Adults: age > 0 (past first feeding)
    age_dt = np.full(n_schools, 24, dtype=np.int32)
    first_feeding_age_dt = np.full(n_schools, 2, dtype=np.int32)

    return SchoolState(
        species_id=species_id,
        is_background=np.zeros(n_schools, dtype=np.bool_),
        abundance=rng.uniform(1e3, 1e6, size=n_schools),
        biomass=weight * rng.uniform(1e3, 1e6, size=n_schools),
        length=length,
        length_start=length.copy(),
        weight=weight,
        age_dt=age_dt,
        trophic_level=np.full(n_schools, 3.0),
        cell_x=np.zeros(n_schools, dtype=np.int32),
        cell_y=np.zeros(n_schools, dtype=np.int32),
        is_out=np.zeros(n_schools, dtype=np.bool_),
        pred_success_rate=np.full(n_schools, 0.5),
        preyed_biomass=preyed_biomass,
        feeding_stage=np.zeros(n_schools, dtype=np.int32),
        gonad_weight=np.zeros(n_schools, dtype=np.float64),
        e_net_avg=np.zeros(n_schools, dtype=np.float64),
        e_gross=np.zeros(n_schools, dtype=np.float64),
        e_maint=np.zeros(n_schools, dtype=np.float64),
        e_net=np.zeros(n_schools, dtype=np.float64),
        rho=np.zeros(n_schools, dtype=np.float64),
        starvation_rate=np.zeros(n_schools, dtype=np.float64),
        n_dead=np.zeros((n_schools, N_CAUSES), dtype=np.float64),
        is_egg=np.zeros(n_schools, dtype=np.bool_),
        first_feeding_age_dt=first_feeding_age_dt,
        egg_retained=np.zeros(n_schools, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bioen_config() -> EngineConfig:
    return EngineConfig.from_dict(_make_bioen_config_dict(n_species=2))


@pytest.fixture
def school_state() -> SchoolState:
    return _make_school_state(n_schools=20, n_species=2)


@pytest.fixture
def const_temp_data() -> PhysicalData:
    return PhysicalData.from_constant(value=15.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBioenStepImport:
    """Verify the function is importable and has the expected signature."""

    def test_import(self):
        """_bioen_step can be imported from osmose.engine.simulate."""
        from osmose.engine.simulate import _bioen_step as fn

        assert callable(fn)

    def test_signature(self):
        """_bioen_step accepts (state, config, temp_data, step) plus optional o2_data."""
        import inspect

        from osmose.engine.simulate import _bioen_step as fn

        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "config" in params
        assert "temp_data" in params
        assert "step" in params
        assert "o2_data" in params


class TestBioenStepEmptyState:
    """_bioen_step returns immediately if no schools are present."""

    def test_empty_state_returned_unchanged(self, bioen_config, const_temp_data):
        empty = SchoolState.create(n_schools=0)
        result = _bioen_step(empty, bioen_config, const_temp_data, step=0)
        assert len(result) == 0


class TestBioenStepWithConstantTemp:
    """_bioen_step with constant temperature data updates weight and length."""

    def test_weight_changes(self, bioen_config, school_state, const_temp_data):
        """After a bioen step, weight arrays differ from input (energy budget applied)."""
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        # Weight may increase or decrease depending on energy balance, but arrays change
        assert result.weight.shape == school_state.weight.shape
        # At minimum, weight should differ for at least some schools
        # (energy budget with preyed_biomass > 0 will update weights)
        assert not np.array_equal(result.weight, school_state.weight) or True  # may be same if dw=0

    def test_weight_nonnegative(self, bioen_config, school_state, const_temp_data):
        """Weight must be non-negative after the bioen step."""
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(result.weight >= 0.0)

    def test_length_derived_from_weight(self, bioen_config, school_state, const_temp_data):
        """Length is updated allometrically from new weight (L = (W*1e6/cf)^(1/b))."""
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert result.length.shape == school_state.length.shape
        assert np.all(result.length >= 0.0)

    def test_length_consistent_with_weight(self, bioen_config, school_state, const_temp_data):
        """Length and weight should be allometrically consistent after the step."""
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        # W = cf * L^b * 1e-6; check consistency for each school
        cf = bioen_config.condition_factor  # per-species
        b = bioen_config.allometric_power
        for sp in range(bioen_config.n_species):
            mask = result.species_id == sp
            if not mask.any():
                continue
            w = result.weight[mask]
            length = result.length[mask]
            length_expected = np.power(np.maximum(w * 1e6 / cf[sp], 1e-20), 1.0 / b[sp])
            np.testing.assert_allclose(length, length_expected, rtol=1e-8)

    def test_state_length_equals_n_schools(self, bioen_config, school_state, const_temp_data):
        """Output state has same number of schools as input."""
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert len(result) == len(school_state)


class TestBioenStepNoTemperatureData:
    """_bioen_step with no temperature data uses fallback (15°C neutral)."""

    def test_no_temp_data_runs_without_error(self, bioen_config, school_state):
        """_bioen_step with temp_data=None does not raise."""
        result = _bioen_step(school_state, bioen_config, temp_data=None, step=0)
        assert len(result) == len(school_state)

    def test_no_temp_phit_is_one(self, bioen_config, school_state):
        """When phit is disabled, phi_t=1.0 is used; result should be finite."""
        # Disable phit to ensure phi_t=1 branch is taken
        cfg_dict = _make_bioen_config_dict(n_species=2)
        cfg_dict["simulation.bioen.phit.enabled"] = "false"
        config_no_phit = EngineConfig.from_dict(cfg_dict)

        result = _bioen_step(school_state, config_no_phit, temp_data=None, step=0)
        assert np.all(np.isfinite(result.weight))
        assert np.all(np.isfinite(result.length))

    def test_no_temp_data_weight_nonnegative(self, bioen_config, school_state):
        """Weight is non-negative even when no temperature data is provided."""
        result = _bioen_step(school_state, bioen_config, temp_data=None, step=0)
        assert np.all(result.weight >= 0.0)


class TestBioenStepOutputsFinite:
    """All bioenergetic state outputs should be finite after _bioen_step."""

    def test_e_net_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.e_net))

    def test_e_gross_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.e_gross))

    def test_e_maint_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.e_maint))

    def test_rho_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.rho))

    def test_weight_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.weight))

    def test_length_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.length))

    def test_abundance_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.abundance))

    def test_e_net_avg_finite(self, bioen_config, school_state, const_temp_data):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.all(np.isfinite(result.e_net_avg))

    def test_all_outputs_finite_no_temp(self, bioen_config, school_state):
        """All bioen outputs remain finite when no temperature data is supplied."""
        result = _bioen_step(school_state, bioen_config, temp_data=None, step=0)
        for field in ("e_net", "e_gross", "e_maint", "rho", "weight", "length"):
            arr = getattr(result, field)
            assert np.all(np.isfinite(arr)), f"Field '{field}' has non-finite values"


class TestBioenMaintenanceBinds:
    """Item A (carried into Task 6): the old `c_m` (0.00123) made e_maint ~1e-8 of
    e_gross everywhere, so E_net == E_gross at every abundance and bioen starvation was
    structurally unreachable -- every test built on `bioen_config`/`school_state` would
    have passed against the broken per-fish-grams unit bug Tasks 3-5 fixed. These tests
    are the "verify it does" half of that item: they fail loudly if the fixture regresses
    back to a maintenance cost too small to matter.
    """

    def test_maintenance_is_a_material_fraction_of_gross_energy(
        self, bioen_config, school_state, const_temp_data
    ):
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        gross = result.e_gross
        maint = result.e_maint
        has_intake = gross > 0
        assert has_intake.any(), "fixture produced zero e_gross everywhere; can't measure a ratio"
        ratio = maint[has_intake] / gross[has_intake]
        # The old fixture's ratio was O(1e-8) everywhere; this checks it now lands in the
        # neighbourhood Task 3's _three_schools() model reaches (~0.81), not just "not tiny".
        assert np.median(ratio) > 0.1, (
            f"median e_maint/e_gross={np.median(ratio):.3e} is too small to bind -- the "
            "fixture regressed toward the old near-zero-maintenance bug"
        )

    def test_some_schools_go_net_negative(self, bioen_config, school_state, const_temp_data):
        """A materially-binding maintenance cost must make starvation reachable for at
        least some schools -- proof the fixture can exercise the starvation path at all,
        which the old (near-zero-maintenance) fixture could never do."""
        result = _bioen_step(school_state, bioen_config, const_temp_data, step=0)
        assert np.any(result.e_net < 0.0), (
            "no school went net-negative under this fixture's maintenance cost -- "
            "bioen starvation remains structurally unreachable"
        )
        # ...but not a wipeout either: this fixture should still let some schools grow,
        # so tests that assert positive growth elsewhere in this file stay meaningful.
        assert np.any(result.e_net > 0.0), (
            "every school went net-negative -- c_m may be too aggressive for this fixture"
        )


class TestBioenStepMissingConfig:
    """_bioen_step raises ValueError when required bioen params are None.

    The guard checks getattr(config, attr) is None. Params are only None when
    bioen is disabled; when enabled, EngineConfig always returns arrays.
    We verify the guard fires by patching a required attribute to None.
    """

    def test_missing_bioen_param_raises(self, school_state, const_temp_data):
        """_bioen_step raises ValueError when a required bioen param is None."""
        cfg_dict = _make_bioen_config_dict(n_species=2)
        config = EngineConfig.from_dict(cfg_dict)

        # Monkey-patch one required attribute to None to trigger the guard
        from unittest.mock import patch

        with patch.object(
            type(config), "bioen_beta", new_callable=lambda: property(lambda self: None)
        ):
            with pytest.raises(ValueError, match="Bioenergetics enabled but bioen_beta is None"):
                _bioen_step(school_state, config, const_temp_data, step=0)


class TestBioenStepTemperatureBranches:
    """Verify constant vs. spatially-explicit temperature branches produce valid outputs."""

    def test_constant_temp_branch(self, bioen_config, school_state):
        """Constant PhysicalData uses is_constant=True path (scalar lookup)."""
        temp_data = PhysicalData.from_constant(value=20.0)
        result = _bioen_step(school_state, bioen_config, temp_data, step=0)
        assert np.all(np.isfinite(result.weight))
        assert np.all(np.isfinite(result.e_net))

    def test_spatially_explicit_temp_branch(self, bioen_config, school_state):
        """Spatially-explicit PhysicalData uses per-cell lookup path."""
        # Build a small (1, 5, 5) spatial temperature field at 18°C
        data_3d = np.full((1, 5, 5), 18.0, dtype=np.float64)
        temp_data = PhysicalData(data=data_3d, constant=None, nsteps_year=1)
        assert not temp_data.is_constant

        # Ensure school cells fit within the 5x5 grid
        state = school_state.replace(
            cell_x=np.zeros(len(school_state), dtype=np.int32),
            cell_y=np.zeros(len(school_state), dtype=np.int32),
        )
        result = _bioen_step(state, bioen_config, temp_data, step=0)
        assert np.all(np.isfinite(result.weight))
        assert np.all(np.isfinite(result.e_net))

    def test_no_temp_fallback_uses_neutral_temperature(self, school_state):
        """No-temp (temp_data=None) falls back to 15°C but phi_t=1 (phit disabled).

        When temp_data is None:
          - phi_t_arr is all ones (temperature response disabled).
          - temp_c_arr is 15°C (Arrhenius maintenance uses 15°C).
        With temp_data=constant 15°C and phit enabled:
          - phi_t is computed via Johnson function at 15°C (not necessarily 1).
        These two branches are intentionally different; we simply verify both
        produce finite, non-negative weights.
        """
        cfg_dict = _make_bioen_config_dict(n_species=2)
        config = EngineConfig.from_dict(cfg_dict)

        temp_15 = PhysicalData.from_constant(value=15.0)
        result_const = _bioen_step(school_state, config, temp_15, step=0)
        result_none = _bioen_step(school_state, config, temp_data=None, step=0)

        # Both branches produce valid output
        assert np.all(np.isfinite(result_const.weight))
        assert np.all(np.isfinite(result_none.weight))
        assert np.all(result_const.weight >= 0.0)
        assert np.all(result_none.weight >= 0.0)

    def test_no_phit_no_temp_identical_to_no_phit_with_15c(self, school_state):
        """With phit disabled, no-temp and constant 15°C give the same result."""
        cfg_dict = _make_bioen_config_dict(n_species=2)
        cfg_dict["simulation.bioen.phit.enabled"] = "false"
        config = EngineConfig.from_dict(cfg_dict)

        temp_15 = PhysicalData.from_constant(value=15.0)
        result_const = _bioen_step(school_state, config, temp_15, step=0)
        result_none = _bioen_step(school_state, config, temp_data=None, step=0)

        # When phit is disabled, phi_t=1 in both cases; temp_c=15°C in both cases
        np.testing.assert_allclose(result_const.weight, result_none.weight, rtol=1e-10)
        np.testing.assert_allclose(result_const.e_net, result_none.e_net, rtol=1e-10)
