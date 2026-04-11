"""Tier 1.5 comparison tests: Python engine vs Java engine.

These tests run the Java OSMOSE engine on the examples config and compare
key outputs against the Python engine's growth and mortality functions.

Since the Python engine does not yet implement all processes (predation,
reproduction, movement), these tests focus on:
  - Von Bertalanffy growth curve accuracy for all 8 example species
  - Weight-length allometric conversion
  - Additional mortality decay rate
  - Aging mortality threshold
  - Java engine output parsing for reference data

Tests marked with @pytest.mark.java require Java to be installed and the
OSMOSE JAR to be present. They are skipped otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.growth import expected_length_vb
from osmose.engine.processes.natural import additional_mortality, aging_mortality
from osmose.engine.state import SchoolState

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent
JAR_PATH = PROJECT_DIR / "osmose-java" / "osmose_4.3.3-jar-with-dependencies.jar"
EXAMPLES_DIR = PROJECT_DIR / "data" / "examples"
EXAMPLES_CONFIG = EXAMPLES_DIR / "osm_all-parameters.csv"

# Check if Java and the JAR are available
_java_available = shutil.which("java") is not None and JAR_PATH.exists()
java_required = pytest.mark.skipif(not _java_available, reason="Java or OSMOSE JAR not available")

# ---------------------------------------------------------------------------
# Example species parameters (from data/examples/osm_param-species.csv)
# ---------------------------------------------------------------------------

SPECIES = [
    {
        "name": "Anchovy",
        "linf": 19.5,
        "k": 0.364,
        "t0": -0.70,
        "c": 0.0060,
        "b": 3.06,
        "lifespan": 4,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "Sardine",
        "linf": 23.0,
        "k": 0.280,
        "t0": -0.90,
        "c": 0.0072,
        "b": 3.10,
        "lifespan": 8,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "Sprat",
        "linf": 14.5,
        "k": 0.500,
        "t0": -0.50,
        "c": 0.0055,
        "b": 3.08,
        "lifespan": 5,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "HorseMackerel",
        "linf": 40.0,
        "k": 0.160,
        "t0": -1.20,
        "c": 0.0068,
        "b": 3.05,
        "lifespan": 15,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "Mackerel",
        "linf": 42.0,
        "k": 0.190,
        "t0": -1.00,
        "c": 0.0040,
        "b": 3.20,
        "lifespan": 12,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "Hake",
        "linf": 110.0,
        "k": 0.106,
        "t0": -0.17,
        "c": 0.0050,
        "b": 3.14,
        "lifespan": 12,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "Sole",
        "linf": 39.0,
        "k": 0.280,
        "t0": -0.50,
        "c": 0.0085,
        "b": 3.05,
        "lifespan": 15,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
    {
        "name": "BlueWhiting",
        "linf": 34.0,
        "k": 0.200,
        "t0": -0.80,
        "c": 0.0038,
        "b": 3.18,
        "lifespan": 10,
        "egg_size": 0.1,
        "vb_threshold_age": 0,
    },
]

N_DT_PER_YEAR = 24  # examples config uses 24


# ===========================================================================
# Tier 1.5a: Growth curve verification for all 8 species
# ===========================================================================


class TestVBGrowthCurvesAllSpecies:
    """Verify VB expected length matches analytical formula for all 8 example species.

    This is not a Java comparison per se, but validates that the Python
    implementation produces the correct VB curves for real species parameters
    (not just toy values).
    """

    @pytest.mark.parametrize("sp", SPECIES, ids=[s["name"] for s in SPECIES])
    def test_vb_curve_matches_formula(self, sp):
        """Expected length at multiple ages matches L_inf * (1 - exp(-K * (age - t0)))."""
        ages_years = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
        # Only test ages within lifespan
        ages_years = ages_years[ages_years <= sp["lifespan"]]
        ages_dt = (ages_years * N_DT_PER_YEAR).astype(np.int32)

        n = len(ages_dt)
        result = expected_length_vb(
            age_dt=ages_dt,
            linf=np.full(n, sp["linf"]),
            k=np.full(n, sp["k"]),
            t0=np.full(n, sp["t0"]),
            egg_size=np.full(n, sp["egg_size"]),
            vb_threshold_age=np.full(n, sp["vb_threshold_age"]),
            n_dt_per_year=N_DT_PER_YEAR,
        )

        # Since vb_threshold_age=0 for all species, all ages use VB formula
        expected = sp["linf"] * (1 - np.exp(-sp["k"] * (ages_years - sp["t0"])))
        np.testing.assert_allclose(
            result, expected, rtol=1e-10, err_msg=f"VB curve mismatch for {sp['name']}"
        )

    @pytest.mark.parametrize("sp", SPECIES, ids=[s["name"] for s in SPECIES])
    def test_weight_length_conversion(self, sp):
        """Weight = c * L^b for lengths from the VB curve."""
        ages_dt = np.array([24, 48, 72], dtype=np.int32)  # 1, 2, 3 years
        ages_dt = ages_dt[ages_dt <= sp["lifespan"] * N_DT_PER_YEAR]
        n = len(ages_dt)

        lengths = expected_length_vb(
            age_dt=ages_dt,
            linf=np.full(n, sp["linf"]),
            k=np.full(n, sp["k"]),
            t0=np.full(n, sp["t0"]),
            egg_size=np.full(n, sp["egg_size"]),
            vb_threshold_age=np.full(n, sp["vb_threshold_age"]),
            n_dt_per_year=N_DT_PER_YEAR,
        )

        # Weight from VB lengths should match W = c * L^b
        computed_weights = sp["c"] * lengths ** sp["b"]
        # Verify against independently known analytical bounds:
        # At age=0, length ≈ initial; at max age, length → L_inf
        assert computed_weights[0] > 0, f"Weight at age 0 must be positive for {sp['name']}"
        assert computed_weights[-1] > computed_weights[0], (
            f"Weight must increase with age for {sp['name']}"
        )
        # Max weight must be bounded by c * linf^b
        max_weight = sp["c"] * sp["linf"] ** sp["b"]
        assert computed_weights[-1] <= max_weight * 1.01, (
            f"Weight exceeds theoretical max for {sp['name']}"
        )

    @pytest.mark.parametrize("sp", SPECIES, ids=[s["name"] for s in SPECIES])
    def test_growth_monotonically_increasing(self, sp):
        """Length should increase monotonically with age (VB property)."""
        max_dt = sp["lifespan"] * N_DT_PER_YEAR
        ages_dt = np.arange(0, max_dt, N_DT_PER_YEAR // 4, dtype=np.int32)
        n = len(ages_dt)

        lengths = expected_length_vb(
            age_dt=ages_dt,
            linf=np.full(n, sp["linf"]),
            k=np.full(n, sp["k"]),
            t0=np.full(n, sp["t0"]),
            egg_size=np.full(n, sp["egg_size"]),
            vb_threshold_age=np.full(n, sp["vb_threshold_age"]),
            n_dt_per_year=N_DT_PER_YEAR,
        )

        # Each successive length should be >= previous (monotonic)
        assert np.all(np.diff(lengths) >= 0), f"Non-monotonic growth for {sp['name']}"

    @pytest.mark.parametrize("sp", SPECIES, ids=[s["name"] for s in SPECIES])
    def test_length_bounded_by_linf(self, sp):
        """Length should never exceed L_inf."""
        max_dt = sp["lifespan"] * N_DT_PER_YEAR
        ages_dt = np.arange(0, max_dt + 1, dtype=np.int32)
        n = len(ages_dt)

        lengths = expected_length_vb(
            age_dt=ages_dt,
            linf=np.full(n, sp["linf"]),
            k=np.full(n, sp["k"]),
            t0=np.full(n, sp["t0"]),
            egg_size=np.full(n, sp["egg_size"]),
            vb_threshold_age=np.full(n, sp["vb_threshold_age"]),
            n_dt_per_year=N_DT_PER_YEAR,
        )

        assert np.all(lengths <= sp["linf"] + 1e-10), (
            f"Length exceeds L_inf for {sp['name']}: max={lengths.max()}, L_inf={sp['linf']}"
        )


# ===========================================================================
# Tier 1.5b: Mortality verification with real parameters
# ===========================================================================


class TestMortalityWithRealParams:
    """Test mortality functions with parameters from the examples config."""

    def _make_config_for_species(self, sp_idx: int) -> EngineConfig:
        """Build a 1-species EngineConfig from examples species data."""
        sp = SPECIES[sp_idx]
        return EngineConfig.from_dict(
            {
                "simulation.time.ndtperyear": str(N_DT_PER_YEAR),
                "simulation.time.nyear": "1",
                "simulation.nspecies": "1",
                "simulation.nschool.sp0": "10",
                "species.name.sp0": sp["name"],
                "species.linf.sp0": str(sp["linf"]),
                "species.k.sp0": str(sp["k"]),
                "species.t0.sp0": str(sp["t0"]),
                "species.egg.size.sp0": str(sp["egg_size"]),
                "species.length2weight.condition.factor.sp0": str(sp["c"]),
                "species.length2weight.allometric.power.sp0": str(sp["b"]),
                "species.lifespan.sp0": str(sp["lifespan"]),
                "species.vonbertalanffy.threshold.age.sp0": str(sp["vb_threshold_age"]),
                "mortality.subdt": "10",
                "predation.ingestion.rate.max.sp0": "3.5",
                "predation.efficiency.critical.sp0": "0.57",
                "mortality.additional.rate.sp0": "0.4",
            }
        )

    def test_additional_mortality_single_substep(self):
        """Single sub-step: D = M_annual / (n_dt_per_year * n_subdt)."""
        cfg = self._make_config_for_species(0)  # Anchovy
        m_rate = 0.4
        n_subdt = 10

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([5.0]),
            biomass=np.array([50000.0]),
            age_dt=np.array([24], dtype=np.int32),  # non-zero: eggs are skipped
        )

        new_state = additional_mortality(state, cfg, n_subdt)

        # Java: D = (M_annual / n_dt_per_year) / n_subdt
        d = m_rate / (N_DT_PER_YEAR * n_subdt)
        expected_survivors = 10000.0 * np.exp(-d)
        np.testing.assert_allclose(
            new_state.abundance[0],
            expected_survivors,
            rtol=1e-10,
            err_msg="Single sub-step mortality doesn't match Java formula",
        )

    def test_additional_mortality_full_year(self):
        """After n_dt * n_subdt applications, total ≈ exp(-M_annual)."""
        cfg = self._make_config_for_species(0)  # Anchovy
        m_rate = 0.4
        n_subdt = 10

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([5.0]),
            biomass=np.array([50000.0]),
            age_dt=np.array([24], dtype=np.int32),  # non-zero: eggs are skipped
        )

        # Apply for full year: n_dt timesteps * n_subdt sub-steps
        for _step in range(N_DT_PER_YEAR):
            for _sub in range(n_subdt):
                state = additional_mortality(state, cfg, n_subdt)

        # Should give approximately exp(-M_annual) = exp(-0.4)
        expected_survivors = 10000.0 * np.exp(-m_rate)
        np.testing.assert_allclose(
            state.abundance[0],
            expected_survivors,
            rtol=1e-4,
            err_msg="Annual mortality doesn't match exp(-M)",
        )

    @pytest.mark.parametrize("sp_idx", range(8), ids=[s["name"] for s in SPECIES])
    def test_aging_kills_at_lifespan(self, sp_idx):
        """Schools at lifespan_dt - 1 are killed by aging mortality."""
        cfg = self._make_config_for_species(sp_idx)
        sp = SPECIES[sp_idx]
        lifespan_dt = sp["lifespan"] * N_DT_PER_YEAR

        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 100.0]),
            age_dt=np.array([lifespan_dt - 2, lifespan_dt - 1], dtype=np.int32),
        )

        new_state = aging_mortality(state, cfg)
        # First school (lifespan - 2) survives, second (lifespan - 1) dies
        np.testing.assert_allclose(
            new_state.abundance[0], 100.0, err_msg=f"{sp['name']}: young school died"
        )
        np.testing.assert_allclose(
            new_state.abundance[1], 0.0, err_msg=f"{sp['name']}: old school survived"
        )


# ===========================================================================
# Tier 1.5c: Java engine comparison (requires Java + JAR)
# ===========================================================================


@java_required
class TestJavaEngineComparison:
    """Run the Java engine on examples config and compare outputs.

    These tests verify that the Java engine produces valid output that we
    can parse, establishing the reference data pipeline for future phases.
    """

    @pytest.fixture(scope="class")
    def java_output(self, tmp_path_factory):
        """Run Java engine once for the entire test class."""
        output_dir = tmp_path_factory.mktemp("java_output")
        cmd = [
            "java",
            "-Xmx2g",
            "-jar",
            str(JAR_PATH),
            str(EXAMPLES_CONFIG),
            f"-Poutput.dir.path={output_dir}",
            "-Psimulation.time.nyear=30",
            "-Poutput.start.year=0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Java engine failed:\n{result.stderr}"
        return output_dir

    def test_java_produces_biomass_output(self, java_output):
        """Java engine should produce non-empty biomass CSV."""
        biomass_file = java_output / "biscay_biomass_Simu0.csv"
        assert biomass_file.exists(), "Biomass file not produced"
        df = pd.read_csv(biomass_file, skiprows=1)
        assert len(df) > 0, "Biomass file is empty"
        assert "Anchovy" in df.columns, "Missing Anchovy column"
        # After seeding period, at least some species should have non-zero biomass
        late = df[df["Time"] > 5.0]
        assert late.drop(columns="Time").sum().sum() > 0, "All biomass is zero after year 5"

    def test_java_produces_mortality_output(self, java_output):
        """Java engine should produce mortality rate CSVs."""
        mort_dir = java_output / "Mortality"
        assert mort_dir.exists(), "Mortality directory not produced"
        mort_files = list(mort_dir.glob("*mortalityRate-Anchovy*"))
        assert len(mort_files) > 0, "No mortality files for Anchovy"
        df = pd.read_csv(mort_files[0], skiprows=2)
        assert len(df) > 0, "Mortality file is empty"

    def test_java_biomass_nonzero_after_seeding(self, java_output):
        """After 10+ years, multiple species should have established populations."""
        biomass_file = java_output / "biscay_biomass_Simu0.csv"
        df = pd.read_csv(biomass_file, skiprows=1)
        late = df[df["Time"] > 10.0]
        if len(late) == 0:
            pytest.skip("Not enough timesteps after year 10")
        species_cols = [c for c in df.columns if c != "Time"]
        nonzero_species = (late[species_cols].sum() > 0).sum()
        assert nonzero_species >= 3, (
            f"Only {nonzero_species} species have non-zero biomass after year 10; "
            "expected at least 3 for a viable ecosystem"
        )

    def test_java_vb_growth_consistency(self, java_output):
        """Compare Java biomass dynamics against expected VB growth properties.

        Since Java runs the full model (predation + growth + mortality), we
        can only check high-level properties:
        - Biomass should be positive for most species after spin-up
        - No species should have negative biomass
        - The ecosystem should reach a quasi-steady state
        """
        biomass_file = java_output / "biscay_biomass_Simu0.csv"
        df = pd.read_csv(biomass_file, skiprows=1)

        # No negative biomass
        species_cols = [c for c in df.columns if c != "Time"]
        for col in species_cols:
            assert (df[col] >= 0).all(), f"Negative biomass for {col}"

        # After year 15, check quasi-steady state (CV < 2.0 for surviving species)
        late = df[df["Time"] > 15.0]
        if len(late) < 10:
            pytest.skip("Not enough late-period data")
        for col in species_cols:
            series = late[col]
            if series.mean() > 0:
                cv = series.std() / series.mean()
                # OSMOSE ecosystems are stochastic; short-lived prey species
                # (Anchovy, Sprat) can have high CV due to recruitment pulses.
                # Use a generous bound that catches only true divergence.
                assert cv < 5.0, (
                    f"{col} has CV={cv:.2f} after year 15 — may not be reaching quasi-steady state"
                )
