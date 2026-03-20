"""Tests for age/size distribution binning in StepOutput (_collect_outputs)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.simulate import _collect_outputs
from osmose.engine.state import SchoolState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    n_species: int = 2,
    n_dt_per_year: int = 12,
    lifespan_years: list[float] | None = None,
    output_biomass_byage: bool = False,
    output_abundance_byage: bool = False,
    output_biomass_bysize: bool = False,
    output_abundance_bysize: bool = False,
    size_min: float = 0.0,
    size_max: float = 40.0,
    size_incr: float = 10.0,
) -> EngineConfig:
    """Build a minimal EngineConfig with distribution output flags."""
    if lifespan_years is None:
        lifespan_years = [3.0] * n_species

    base: dict[str, str] = {
        "simulation.nspecies": str(n_species),
        "simulation.time.ndtperyear": str(n_dt_per_year),
        "simulation.time.nyear": "1",
        "mortality.subdt": "10",
    }
    for i in range(n_species):
        base[f"species.name.sp{i}"] = f"Sp{i}"
        base[f"species.linf.sp{i}"] = "20.0"
        base[f"species.k.sp{i}"] = "0.3"
        base[f"species.t0.sp{i}"] = "-0.1"
        base[f"species.egg.size.sp{i}"] = "0.1"
        base[f"species.length2weight.condition.factor.sp{i}"] = "0.006"
        base[f"species.length2weight.allometric.power.sp{i}"] = "3.0"
        base[f"species.lifespan.sp{i}"] = str(lifespan_years[i])
        base[f"species.vonbertalanffy.threshold.age.sp{i}"] = "1.0"
        base[f"predation.ingestion.rate.max.sp{i}"] = "3.5"
        base[f"predation.efficiency.critical.sp{i}"] = "0.57"

    if output_biomass_byage:
        base["output.biomass.byage.enabled"] = "true"
    if output_abundance_byage:
        base["output.abundance.byage.enabled"] = "true"
    if output_biomass_bysize:
        base["output.biomass.bysize.enabled"] = "true"
    if output_abundance_bysize:
        base["output.abundance.bysize.enabled"] = "true"

    base["output.distrib.bysize.min"] = str(size_min)
    base["output.distrib.bysize.max"] = str(size_max)
    base["output.distrib.bysize.incr"] = str(size_incr)

    return EngineConfig.from_dict(base)


def _make_state(
    species_ids: list[int],
    age_dts: list[int],
    lengths: list[float],
    biomass: list[float],
    abundance: list[float] | None = None,
) -> SchoolState:
    """Build a minimal SchoolState for testing."""
    n = len(species_ids)
    if abundance is None:
        abundance = [1.0] * n
    sp_arr = np.array(species_ids, dtype=np.int32)
    state = SchoolState.create(n_schools=n, species_id=sp_arr)
    state = state.replace(
        age_dt=np.array(age_dts, dtype=np.int32),
        length=np.array(lengths, dtype=np.float64),
        biomass=np.array(biomass, dtype=np.float64),
        abundance=np.array(abundance, dtype=np.float64),
    )
    return state


# ---------------------------------------------------------------------------
# Age binning tests
# ---------------------------------------------------------------------------


class TestAgeBinning:
    def test_biomass_by_age_bins(self):
        """Schools at different ages land in correct age bins."""
        cfg = _make_config(
            n_species=2,
            n_dt_per_year=12,
            lifespan_years=[3.0, 3.0],
            output_biomass_byage=True,
        )
        # sp0: school at age 0 dt (age_yr=0), school at age 12 dt (age_yr=1)
        state = _make_state(
            species_ids=[0, 0],
            age_dts=[0, 12],
            lengths=[1.0, 5.0],
            biomass=[100.0, 200.0],
        )
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_age is not None
        bba = out.biomass_by_age[0]
        assert bba[0] == pytest.approx(100.0)
        assert bba[1] == pytest.approx(200.0)
        assert bba[2] == pytest.approx(0.0)

    def test_species_1_age_bins(self):
        """Second species ages binned correctly."""
        cfg = _make_config(
            n_species=2,
            n_dt_per_year=12,
            lifespan_years=[3.0, 5.0],
            output_biomass_byage=True,
        )
        # sp0: one school at age_yr=2; sp1: one school at age_yr=4
        state = _make_state(
            species_ids=[0, 1],
            age_dts=[24, 48],
            lengths=[8.0, 15.0],
            biomass=[50.0, 75.0],
        )
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_age is not None
        assert out.biomass_by_age[0][2] == pytest.approx(50.0)
        assert out.biomass_by_age[1][4] == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Size binning tests
# ---------------------------------------------------------------------------


class TestSizeBinning:
    def test_biomass_by_size_bins(self):
        """Schools at different lengths land in correct size bins."""
        # bins: [0,10), [10,20), [20,30), [30,40)  -> 4 bins with min=0,max=40,incr=10
        cfg = _make_config(
            n_species=1,
            n_dt_per_year=12,
            lifespan_years=[3.0],
            output_biomass_bysize=True,
            size_min=0.0,
            size_max=40.0,
            size_incr=10.0,
        )
        # lengths: 5 -> bin 0, 15 -> bin 1, 25 -> bin 2
        state = _make_state(
            species_ids=[0, 0, 0],
            age_dts=[0, 12, 24],
            lengths=[5.0, 15.0, 25.0],
            biomass=[10.0, 20.0, 30.0],
        )
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_size is not None
        bbs = out.biomass_by_size[0]
        assert bbs[0] == pytest.approx(10.0)
        assert bbs[1] == pytest.approx(20.0)
        assert bbs[2] == pytest.approx(30.0)

    def test_size_at_bin_boundary(self):
        """Length exactly at bin edge is placed in the next bin (right-exclusive)."""
        cfg = _make_config(
            n_species=1,
            n_dt_per_year=12,
            lifespan_years=[3.0],
            output_biomass_bysize=True,
            size_min=0.0,
            size_max=40.0,
            size_incr=10.0,
        )
        # length=10.0 exactly: searchsorted(edges=[0,10,20,30,40], 10, side='right')=2, bin=1
        state = _make_state(
            species_ids=[0],
            age_dts=[0],
            lengths=[10.0],
            biomass=[50.0],
        )
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_size is not None
        bbs = out.biomass_by_size[0]
        # bin 1 should have the biomass (10.0 is in [10,20))
        assert bbs[1] == pytest.approx(50.0)
        assert bbs[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestDistributionEdgeCases:
    def test_empty_state_produces_none(self):
        """When disabled, distribution fields are None."""
        cfg = _make_config(n_species=1, n_dt_per_year=12, lifespan_years=[3.0])
        state = SchoolState.create(n_schools=0, species_id=np.array([], dtype=np.int32))
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_age is None
        assert out.abundance_by_age is None
        assert out.biomass_by_size is None
        assert out.abundance_by_size is None

    def test_single_school_single_bin(self):
        """Single school produces one non-zero bin entry."""
        cfg = _make_config(
            n_species=1,
            n_dt_per_year=12,
            lifespan_years=[3.0],
            output_biomass_byage=True,
            output_biomass_bysize=True,
            size_min=0.0,
            size_max=40.0,
            size_incr=10.0,
        )
        state = _make_state(
            species_ids=[0],
            age_dts=[0],
            lengths=[3.0],
            biomass=[42.0],
        )
        out = _collect_outputs(state, cfg, step=5)

        assert out.biomass_by_age is not None
        assert out.biomass_by_size is not None
        # Only one non-zero entry in age bins
        assert out.biomass_by_age[0].sum() == pytest.approx(42.0)
        assert out.biomass_by_age[0][0] == pytest.approx(42.0)
        # Only one non-zero entry in size bins
        assert out.biomass_by_size[0].sum() == pytest.approx(42.0)
        assert out.biomass_by_size[0][0] == pytest.approx(42.0)

    def test_all_same_age_single_bin(self):
        """All schools at same age -> all biomass in one age bin."""
        cfg = _make_config(
            n_species=1,
            n_dt_per_year=12,
            lifespan_years=[3.0],
            output_biomass_byage=True,
        )
        # All 5 schools at age_yr=1 (age_dt=12..23)
        state = _make_state(
            species_ids=[0, 0, 0, 0, 0],
            age_dts=[12, 13, 14, 15, 16],
            lengths=[5.0, 5.0, 5.0, 5.0, 5.0],
            biomass=[10.0, 10.0, 10.0, 10.0, 10.0],
        )
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_age is not None
        bba = out.biomass_by_age[0]
        assert bba[1] == pytest.approx(50.0)
        assert bba[0] == pytest.approx(0.0)
        assert bba[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Disabled tests
# ---------------------------------------------------------------------------


class TestDistributionDisabled:
    def test_disabled_by_default(self):
        """Distribution fields are None when config flags are false."""
        cfg = _make_config(
            n_species=2,
            n_dt_per_year=12,
            lifespan_years=[3.0, 3.0],
            # no output_*_by* flags set -> defaults False
        )
        state = _make_state(
            species_ids=[0, 1],
            age_dts=[5, 10],
            lengths=[3.0, 8.0],
            biomass=[100.0, 200.0],
        )
        out = _collect_outputs(state, cfg, step=0)

        assert out.biomass_by_age is None
        assert out.abundance_by_age is None
        assert out.biomass_by_size is None
        assert out.abundance_by_size is None

    def test_biomass_and_abundance_independent(self):
        """Enabling only biomass_byage leaves abundance_by_age None and vice-versa."""
        cfg_bm = _make_config(
            n_species=1,
            n_dt_per_year=12,
            lifespan_years=[3.0],
            output_biomass_byage=True,
            output_abundance_byage=False,
        )
        cfg_ab = _make_config(
            n_species=1,
            n_dt_per_year=12,
            lifespan_years=[3.0],
            output_biomass_byage=False,
            output_abundance_byage=True,
        )
        state = _make_state(
            species_ids=[0],
            age_dts=[6],
            lengths=[4.0],
            biomass=[50.0],
            abundance=[500.0],
        )

        out_bm = _collect_outputs(state, cfg_bm, step=0)
        assert out_bm.biomass_by_age is not None
        assert out_bm.abundance_by_age is None

        out_ab = _collect_outputs(state, cfg_ab, step=0)
        assert out_ab.biomass_by_age is None
        assert out_ab.abundance_by_age is not None
        assert out_ab.abundance_by_age[0][0] == pytest.approx(500.0)
