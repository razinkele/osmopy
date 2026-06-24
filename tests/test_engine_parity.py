"""Parity gate tests for the Python OSMOSE engine.

Compares current engine output against a saved baseline to ensure
optimizations produce identical results with the same RNG seed.

Generate baselines with:
    .venv/bin/python scripts/save_parity_baseline.py --years 1 --seed 42

Tests are skipped if no baseline file exists (CI should generate one first).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate

PROJECT_DIR = Path(__file__).parent.parent
EXAMPLES_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"
EEC_CONFIG = PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv"
EEC_BASE_DIR = PROJECT_DIR / "data" / "eec_full"
BASELINE_DIR = PROJECT_DIR / "tests" / "baselines"

# Default baseline parameters
DEFAULT_YEARS = 1
DEFAULT_SEED = 42

# The committed .npz baseline reproduces bit-for-bit only within the
# environment that generated it (local CPython 3.12). Exact reproduction breaks
# elsewhere for toolchain/hardware reasons, NOT arithmetic regressions:
#   * CPython 3.13 reorders floating-point ops at the ULP level, and this
#     chaotic engine amplifies that into large trajectory divergence.
#   * On CI, the runner's CPU/SIMD and Python patch level introduce their own
#     ULP-level differences. With the FIE seeding fallback active, bob's
#     trajectory sits near extinction bifurcations, so those ULP differences
#     flip extinctions into large divergence. Verified: the baseline reproduces
#     exactly on this machine under both numba 0.64.0 and 0.65.1, and the full
#     suite passes locally under CI's exact numeric stack, yet the GitHub runner
#     diverges (~55% of biomass cells, max relative diff 1.0).
# So exact-match runs only locally on the baseline interpreter, as a detector
# for *intentional* arithmetic changes (regenerate the baseline when it fires;
# see scripts/save_parity_baseline.py). Portable parity is enforced everywhere
# by TestStatisticalParity (10-seed mean within tolerance) plus the shape checks.
_BASELINE_PYTHON = (3, 12)
_RUNNING_ON_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
_exact_match_local_only = pytest.mark.skipif(
    _RUNNING_ON_CI or sys.version_info[:2] != _BASELINE_PYTHON,
    reason=(
        "Bit-exact parity reproduces only within its generating environment "
        f"(local CPython {_BASELINE_PYTHON[0]}.{_BASELINE_PYTHON[1]}); "
        "portable parity is gated by TestStatisticalParity"
    ),
)


def _baseline_path(n_years: int = DEFAULT_YEARS, seed: int = DEFAULT_SEED) -> Path:
    return BASELINE_DIR / f"parity_baseline_bob_{n_years}yr_seed{seed}.npz"


def _eec_baseline_path(n_years: int = DEFAULT_YEARS, seed: int = DEFAULT_SEED) -> Path:
    return BASELINE_DIR / f"parity_baseline_eec_{n_years}yr_seed{seed}.npz"


def _run_engine(n_years: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the engine and return (biomass, abundance, mortality) arrays."""
    from osmose.config.reader import OsmoseConfigReader

    reader = OsmoseConfigReader()
    raw = reader.read(EXAMPLES_CONFIG)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            PROJECT_DIR / "data" / "examples" / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)
    outputs = simulate(cfg, grid, rng)

    n_steps = len(outputs)
    n_species = len(outputs[0].biomass)
    n_causes = outputs[0].mortality_by_cause.shape[1]

    biomass = np.zeros((n_steps, n_species), dtype=np.float64)
    abundance = np.zeros((n_steps, n_species), dtype=np.float64)
    mortality = np.zeros((n_steps, n_species, n_causes), dtype=np.float64)

    for i, out in enumerate(outputs):
        biomass[i] = out.biomass
        abundance[i] = out.abundance
        mortality[i] = out.mortality_by_cause

    return biomass, abundance, mortality


def _load_baseline(
    n_years: int = DEFAULT_YEARS, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load baseline arrays. Returns (biomass, abundance, mortality, species_names)."""
    path = _baseline_path(n_years, seed)
    data = np.load(path)
    return (
        data["biomass"],
        data["abundance"],
        data["mortality"],
        list(data["species_names"]),
    )


def _run_engine_eec(n_years: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the Python engine on EEC config; return (biomass, abundance, mortality)."""
    from osmose.config.reader import OsmoseConfigReader

    reader = OsmoseConfigReader()
    raw = reader.read(EEC_CONFIG)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            EEC_BASE_DIR / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)
    outputs = simulate(cfg, grid, rng)

    n_steps = len(outputs)
    n_species = len(outputs[0].biomass)
    n_causes = outputs[0].mortality_by_cause.shape[1]

    biomass = np.zeros((n_steps, n_species), dtype=np.float64)
    abundance = np.zeros((n_steps, n_species), dtype=np.float64)
    mortality = np.zeros((n_steps, n_species, n_causes), dtype=np.float64)

    for i, out in enumerate(outputs):
        biomass[i] = out.biomass
        abundance[i] = out.abundance
        mortality[i] = out.mortality_by_cause

    return biomass, abundance, mortality


def _load_eec_baseline(
    n_years: int = DEFAULT_YEARS, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load EEC baseline arrays. Returns (biomass, abundance, mortality, species_names)."""
    path = _eec_baseline_path(n_years, seed)
    data = np.load(path)
    return (
        data["biomass"],
        data["abundance"],
        data["mortality"],
        list(data["species_names"]),
    )


# ---------------------------------------------------------------------------
# Baseline parity tests (exact match — same seed must produce same output)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_and_current():
    """Run the engine once and load the baseline for comparison."""
    path = _baseline_path()
    if not path.exists():
        pytest.skip(f"No baseline file: {path}. Run scripts/save_parity_baseline.py first.")
    if not EXAMPLES_CONFIG.exists():
        pytest.skip(f"No example config: {EXAMPLES_CONFIG}")

    b_bio, b_abd, b_mort, species = _load_baseline()
    c_bio, c_abd, c_mort = _run_engine(DEFAULT_YEARS, DEFAULT_SEED)

    return {
        "baseline_biomass": b_bio,
        "baseline_abundance": b_abd,
        "baseline_mortality": b_mort,
        "current_biomass": c_bio,
        "current_abundance": c_abd,
        "current_mortality": c_mort,
        "species": species,
    }


class TestBaselineParity:
    """Current output must match the baseline within tolerance.

    Arithmetic reorderings (e.g., caching inst_abd) change FP rounding at the
    ULP level, which can cascade through stochastic mortality interactions.
    We use exact match (atol=0) by default; after an intentional arithmetic
    change, regenerate the baseline with scripts/save_parity_baseline.py.
    """

    @_exact_match_local_only
    def test_biomass_match(self, baseline_and_current):
        d = baseline_and_current
        np.testing.assert_array_equal(
            d["current_biomass"],
            d["baseline_biomass"],
            err_msg="Biomass differs from baseline — regenerate if arithmetic changed",
        )

    @_exact_match_local_only
    def test_abundance_match(self, baseline_and_current):
        d = baseline_and_current
        np.testing.assert_array_equal(
            d["current_abundance"],
            d["baseline_abundance"],
            err_msg="Abundance differs from baseline — regenerate if arithmetic changed",
        )

    @_exact_match_local_only
    def test_mortality_match(self, baseline_and_current):
        d = baseline_and_current
        np.testing.assert_array_equal(
            d["current_mortality"],
            d["baseline_mortality"],
            err_msg="Mortality differs from baseline — regenerate if arithmetic changed",
        )

    def test_step_count_matches(self, baseline_and_current):
        d = baseline_and_current
        assert d["current_biomass"].shape[0] == d["baseline_biomass"].shape[0], (
            "Step count changed!"
        )

    def test_species_count_matches(self, baseline_and_current):
        d = baseline_and_current
        assert d["current_biomass"].shape[1] == d["baseline_biomass"].shape[1], (
            "Species count changed!"
        )


# ---------------------------------------------------------------------------
# Determinism tests (no baseline needed)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed must produce identical results across runs."""

    @pytest.mark.skipif(
        not EXAMPLES_CONFIG.exists(), reason="No example config for determinism test"
    )
    def test_same_seed_same_output(self):
        """Two runs with the same seed must produce bit-identical biomass."""
        bio1, _, _ = _run_engine(n_years=1, seed=99)
        bio2, _, _ = _run_engine(n_years=1, seed=99)
        np.testing.assert_array_equal(
            bio1, bio2, err_msg="Same seed produced different outputs — non-determinism detected!"
        )

    @pytest.mark.skipif(
        not EXAMPLES_CONFIG.exists(), reason="No example config for determinism test"
    )
    def test_different_seeds_differ(self):
        """Different seeds should produce different outputs."""
        bio1, _, _ = _run_engine(n_years=1, seed=42)
        bio2, _, _ = _run_engine(n_years=1, seed=123)
        assert not np.array_equal(bio1, bio2), "Different seeds produced identical outputs!"


# ---------------------------------------------------------------------------
# Sanity checks (no baseline needed)
# ---------------------------------------------------------------------------


class TestSanityChecks:
    """Basic invariants that must hold regardless of optimization tier."""

    @pytest.mark.skipif(not EXAMPLES_CONFIG.exists(), reason="No example config")
    def test_biomass_non_negative(self):
        bio, _, _ = _run_engine(n_years=1, seed=42)
        assert np.all(bio >= 0), "Negative biomass detected!"

    @pytest.mark.skipif(not EXAMPLES_CONFIG.exists(), reason="No example config")
    def test_abundance_non_negative(self):
        _, abd, _ = _run_engine(n_years=1, seed=42)
        assert np.all(abd >= 0), "Negative abundance detected!"

    @pytest.mark.skipif(not EXAMPLES_CONFIG.exists(), reason="No example config")
    def test_mortality_non_negative(self):
        _, _, mort = _run_engine(n_years=1, seed=42)
        assert np.all(mort >= 0), "Negative mortality detected!"

    @pytest.mark.skipif(not EXAMPLES_CONFIG.exists(), reason="No example config")
    def test_some_species_survive(self):
        """At least some species should have nonzero biomass at end of year 1."""
        bio, _, _ = _run_engine(n_years=1, seed=42)
        final_biomass = bio[-1]
        assert np.any(final_biomass > 0), "All species went extinct in year 1!"


# ---------------------------------------------------------------------------
# Statistical parity tests (cross-version — tolerate RNG order changes)
# ---------------------------------------------------------------------------

STATISTICAL_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def _statistical_baseline_path(n_years: int = DEFAULT_YEARS, n_seeds: int = 10) -> Path:
    return BASELINE_DIR / f"statistical_baseline_bob_{n_years}yr_{n_seeds}seeds.npz"


class TestStatisticalParity:
    """Mean biomass across 10 seeds must stay within 5% of baseline.

    This test tolerates RNG consumption order changes (e.g., Phase A batch
    cell loop) that produce different per-seed results but statistically
    equivalent distributions.
    """

    @pytest.fixture(scope="class")
    def statistical_data(self):
        path = _statistical_baseline_path()
        if not path.exists():
            pytest.skip(
                f"No statistical baseline: {path}. "
                "Run: scripts/save_parity_baseline.py --statistical"
            )
        if not EXAMPLES_CONFIG.exists():
            pytest.skip(f"No example config: {EXAMPLES_CONFIG}")

        data = np.load(path)
        baseline_means = data["mean_biomass"]

        # Run engine with same seeds
        final_biomasses = []
        for seed in STATISTICAL_SEEDS:
            bio, _, _ = _run_engine(DEFAULT_YEARS, seed)
            final_biomasses.append(bio[-1])

        current_means = np.mean(final_biomasses, axis=0)
        return {"baseline_means": baseline_means, "current_means": current_means}

    def test_multi_seed_biomass_within_tolerance(self, statistical_data):
        """10 seeds, mean final biomass per species within 5% of baseline."""
        np.testing.assert_allclose(
            statistical_data["current_means"],
            statistical_data["baseline_means"],
            rtol=0.05,
            atol=1.0,
            err_msg="Statistical parity failed — mean biomass drifted >5% from baseline",
        )


# ---------------------------------------------------------------------------
# EEC baseline parity tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eec_baseline_and_current():
    """Run the EEC engine once and load the EEC baseline for comparison."""
    path = _eec_baseline_path()
    if not path.exists():
        pytest.skip(
            f"No EEC baseline file: {path}. "
            "Run: scripts/save_parity_baseline.py --config eec --years 1 --seed 42"
        )
    if not EEC_CONFIG.exists():
        pytest.skip(f"No EEC config: {EEC_CONFIG}")

    b_bio, b_abd, b_mort, species = _load_eec_baseline()
    c_bio, c_abd, c_mort = _run_engine_eec(DEFAULT_YEARS, DEFAULT_SEED)

    return {
        "baseline_biomass": b_bio,
        "baseline_abundance": b_abd,
        "baseline_mortality": b_mort,
        "current_biomass": c_bio,
        "current_abundance": c_abd,
        "current_mortality": c_mort,
        "species": species,
    }


class TestEecBaselineParity:
    """EEC config: current output must match the EEC baseline within tolerance.

    Like BoB (TestBaselineParity), exact match is enforced only locally on the
    baseline-generating interpreter; shape and statistical checks are portable.
    Regenerate with: scripts/save_parity_baseline.py --config eec --years 1 --seed 42
    """

    @_exact_match_local_only
    def test_biomass_match(self, eec_baseline_and_current):
        d = eec_baseline_and_current
        np.testing.assert_array_equal(
            d["current_biomass"],
            d["baseline_biomass"],
            err_msg="EEC biomass differs from baseline — regenerate if arithmetic changed",
        )

    @_exact_match_local_only
    def test_abundance_match(self, eec_baseline_and_current):
        d = eec_baseline_and_current
        np.testing.assert_array_equal(
            d["current_abundance"],
            d["baseline_abundance"],
            err_msg="EEC abundance differs from baseline — regenerate if arithmetic changed",
        )

    @_exact_match_local_only
    def test_mortality_match(self, eec_baseline_and_current):
        d = eec_baseline_and_current
        np.testing.assert_array_equal(
            d["current_mortality"],
            d["baseline_mortality"],
            err_msg="EEC mortality differs from baseline — regenerate if arithmetic changed",
        )

    def test_step_count_matches(self, eec_baseline_and_current):
        d = eec_baseline_and_current
        assert d["current_biomass"].shape[0] == d["baseline_biomass"].shape[0], (
            "EEC step count changed!"
        )

    def test_species_count_matches(self, eec_baseline_and_current):
        d = eec_baseline_and_current
        assert d["current_biomass"].shape[1] == d["baseline_biomass"].shape[1], (
            "EEC species count changed!"
        )
