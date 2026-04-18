#!/usr/bin/env python3
"""Baltic Sea OSMOSE calibration script.

Calibrates larval mortality rates (phase 1) and optionally adult mortality / fishing
rates (phase 2) to match target equilibrium biomass for 8 focal species.

Uses the PythonEngine directly (no Java required).

Usage:
    .venv/bin/python scripts/calibrate_baltic.py [--phase 1] [--maxiter 200] [--seeds 3]
    .venv/bin/python scripts/calibrate_baltic.py --phase 2 --maxiter 300
    .venv/bin/python scripts/calibrate_baltic.py --validate  # 50yr validation of best params
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALTIC_CONFIG = PROJECT_ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
TARGETS_CSV = PROJECT_ROOT / "data" / "baltic" / "reference" / "biomass_targets.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "baltic" / "calibration_results"

SPECIES_NAMES = [
    "cod", "herring", "sprat", "flounder",
    "perch", "pikeperch", "whitefish", "stickleback",
]
N_SPECIES = len(SPECIES_NAMES)


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------
@dataclass
class BiomassTarget:
    species: str
    target: float
    lower: float
    upper: float
    weight: float = 1.0


def load_targets(path: Path = TARGETS_CSV) -> list[BiomassTarget]:
    """Load calibration targets from CSV (skips # comment lines)."""
    targets = []
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        targets.append(BiomassTarget(
            species=row["species"],
            target=float(row["target_tonnes"]),
            lower=float(row["lower_tonnes"]),
            upper=float(row["upper_tonnes"]),
            weight=float(row.get("weight", "1.0")),
        ))
    return targets


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------
def run_simulation(
    config: dict[str, str],
    overrides: dict[str, str],
    n_years: int = 40,
    seed: int = 42,
) -> dict[str, float]:
    """Run PythonEngine with overrides, return last-10-year mean biomass per species.

    Also returns stability metrics (CV, trend slope) for penalty computation.
    """
    from osmose.engine import PythonEngine
    from osmose.results import OsmoseResults

    cfg = dict(config)
    cfg["simulation.time.nyear"] = str(n_years)
    cfg.update(overrides)

    with tempfile.TemporaryDirectory(prefix="baltic_cal_") as tmpdir:
        output_dir = Path(tmpdir) / "output"
        engine = PythonEngine()
        result = engine.run(cfg, output_dir=output_dir, seed=seed)

        if result.returncode != 0:
            return {}

        results = OsmoseResults(output_dir, strict=False)
        bio = results.biomass()
        results.close()

    # Extract last 10 years of biomass
    n_eval_years = 10
    total_years = len(bio)
    if total_years <= n_eval_years:
        eval_data = bio
    else:
        eval_data = bio.iloc[-n_eval_years:]

    species_stats: dict[str, float] = {}
    for sp in SPECIES_NAMES:
        if sp in eval_data.columns:
            vals = eval_data[sp].values.astype(float)
            mean_val = float(np.mean(vals))
            species_stats[f"{sp}_mean"] = mean_val

            # CV for stability penalty
            if mean_val > 0:
                species_stats[f"{sp}_cv"] = float(np.std(vals) / mean_val)
            else:
                species_stats[f"{sp}_cv"] = 10.0

            # Linear trend (normalized slope) for stability penalty
            if len(vals) >= 3:
                x = np.arange(len(vals), dtype=float)
                slope = np.polyfit(x, vals, 1)[0]
                species_stats[f"{sp}_trend"] = float(abs(slope) / (mean_val + 1.0))
            else:
                species_stats[f"{sp}_trend"] = 0.0

    return species_stats


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def make_objective(
    base_config: dict[str, str],
    targets: list[BiomassTarget],
    param_keys: list[str],
    n_years: int = 40,
    seed: int = 42,
    use_log_space: bool = True,
    w_stability: float = 5.0,
    w_worst: float = 0.5,
) -> Callable[[np.ndarray], float]:
    """Create objective function for differential evolution.

    Objective components:
      1. Banded log-ratio loss: 0 inside [lower, upper], squared log-distance outside
      2. Species weights: high (1.0) for well-assessed, low (0.2) for uncertain
      3. Worst-species max term: prevents hiding one badly wrong species
      4. Stability penalties: CV > 0.2 and trend > 0.05, weighted by w_stability
    """
    target_dict = {t.species: t for t in targets}
    _calls = count(1)

    def objective(x: np.ndarray) -> float:
        call_idx = next(_calls)

        # Map parameter vector to config overrides
        overrides: dict[str, str] = {}
        for i, key in enumerate(param_keys):
            if use_log_space:
                val = 10.0 ** x[i]
            else:
                val = x[i]
            overrides[key] = str(val)

        # Run simulation
        stats = run_simulation(base_config, overrides, n_years=n_years, seed=seed)
        if not stats:
            return 1e6  # Failed run

        # Compute objective
        total_error = 0.0
        worst_error = 0.0
        for sp in SPECIES_NAMES:
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in stats or sp not in target_dict:
                total_error += 100.0
                worst_error = max(worst_error, 100.0)
                continue

            sim_biomass = stats[mean_key]
            target = target_dict[sp]

            # Banded log-ratio loss: zero within [lower, upper]
            if sim_biomass <= 0:
                sp_error = 100.0
            elif sim_biomass < target.lower:
                sp_error = np.log10(target.lower / sim_biomass) ** 2
            elif sim_biomass > target.upper:
                sp_error = np.log10(sim_biomass / target.upper) ** 2
            else:
                sp_error = 0.0  # Within acceptable range

            # Apply species weight
            weighted_error = target.weight * sp_error
            total_error += weighted_error
            worst_error = max(worst_error, weighted_error)

            # Stability penalty: penalize high CV (oscillations)
            cv = stats.get(cv_key, 0.0)
            if cv > 0.2:
                total_error += w_stability * target.weight * (cv - 0.2) ** 2

            # Trend penalty: penalize non-equilibrium
            trend = stats.get(trend_key, 0.0)
            if trend > 0.05:
                total_error += w_stability * target.weight * (trend - 0.05) ** 2

        # Add worst-species term to prevent hiding one bad species
        total_error += w_worst * worst_error

        if call_idx % 5 == 0:
            biomass_summary = ", ".join(
                f"{sp}={stats.get(f'{sp}_mean', 0):.0f}"
                for sp in SPECIES_NAMES
            )
            print(
                f"  [eval {call_idx}] obj={total_error:.4f} | {biomass_summary}",
                flush=True,
            )

        return total_error

    return objective


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
def get_phase1_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 1: combined larval + adult mortality (16 params, log10 space).

    Food web analysis showed that:
    - Larval mortality controls recruitment (how many new fish enter population)
    - Adult mortality controls persistence (how fast adults are removed)
    - Both are needed simultaneously due to strong predator-prey interactions
    - Ingestion rate changes have counterintuitive effects (size-refuge dynamics)

    R18 starting values are used as initial guess for the optimizer.
    """
    keys = []
    bounds = []
    x0 = []  # R18 starting point in log10 space

    # Larval mortality: R18 values range 0.4-15.5
    # Bounds: log10(0.1)=-1.0 to log10(100)=2.0
    r18_larval = [15.0, 8.0, 9.0, 12.0, 13.0, 15.0, 13.5, 3.5]
    larval_bounds = [
        (-1.0, 2.0),   # sp0 cod
        (-1.0, 2.0),   # sp1 herring — may need very high larval mort
        (-1.0, 2.0),   # sp2 sprat
        (-1.0, 2.0),   # sp3 flounder
        (-1.0, 2.0),   # sp4 perch
        (-1.0, 2.0),   # sp5 pikeperch
        (-1.0, 2.0),   # sp6 whitefish
        (-1.0, 2.0),   # sp7 stickleback
    ]
    for i in range(N_SPECIES):
        keys.append(f"mortality.additional.larva.rate.sp{i}")
        bounds.append(larval_bounds[i])
        x0.append(np.log10(max(r18_larval[i], 0.1)))

    # Adult additional mortality: R18 values range 0.0-0.05
    # Bounds: log10(0.001)=-3.0 to log10(2.0)=0.3
    r18_adult = [0.05, 0.001, 0.05, 0.05, 0.02, 0.03, 0.02, 0.001]
    adult_bounds = [
        (-3.0, 0.3),   # sp0 cod
        (-3.0, 0.3),   # sp1 herring
        (-3.0, 0.3),   # sp2 sprat
        (-3.0, 0.3),   # sp3 flounder
        (-3.0, 0.3),   # sp4 perch
        (-3.0, 0.3),   # sp5 pikeperch
        (-3.0, 0.3),   # sp6 whitefish
        (-3.0, 0.3),   # sp7 stickleback
    ]
    for i in range(N_SPECIES):
        keys.append(f"mortality.additional.rate.sp{i}")
        bounds.append(adult_bounds[i])
        x0.append(np.log10(max(r18_adult[i], 0.001)))

    return keys, bounds, x0


def get_phase1b_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 1b: focused calibration of 4 problematic planktivores + cod ingestion.

    Only herring(sp1), sprat(sp2), flounder(sp3), stickleback(sp7) are far from
    ICES targets at R18 values. The other 4 species are kept at R18 defaults.

    Cod ingestion rate is included as a structural lever because Ryberg et al.
    (2020, doi:10.1093/conphys/coaa093) showed Contracaecum osculatum parasites
    reduce Baltic cod metabolic rate and foraging capacity. This parameter lets
    the optimizer balance cod's predation pressure against planktivore mortality.

    9 params total: 4 larval + 4 adult mortality + 1 cod ingestion rate.
    """
    # Species indices for the 4 problem species
    problem_species = [1, 2, 3, 7]  # herring, sprat, flounder, stickleback

    keys = []
    bounds = []
    x0 = []

    r18_larval = {1: 8.0, 2: 9.0, 3: 12.0, 7: 0.4}
    r18_adult = {1: 0.001, 2: 0.05, 3: 0.05, 7: 0.001}

    for sp_idx in problem_species:
        keys.append(f"mortality.additional.larva.rate.sp{sp_idx}")
        bounds.append((-0.5, 2.0))  # 0.3 to 100
        x0.append(np.log10(max(r18_larval[sp_idx], 0.1)))

    for sp_idx in problem_species:
        keys.append(f"mortality.additional.rate.sp{sp_idx}")
        bounds.append((-2.5, 0.5))  # 0.003 to 3.16
        x0.append(np.log10(max(r18_adult[sp_idx], 0.003)))

    # Cod ingestion rate: parasite-adjusted (Ryberg et al. 2020)
    # R18 default = 3.5, parasites could reduce to ~1.5-3.5
    keys.append("predation.ingestion.rate.max.sp0")
    bounds.append((np.log10(1.5), np.log10(4.0)))  # 1.5 to 4.0
    x0.append(np.log10(3.5))  # start at R18 default

    return keys, bounds, x0


def get_phase1c_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 1c: Phase 1b + LTL plankton accessibility controls.

    Adds 4 plankton accessibility parameters (diatoms, microzoo, mesozoo, macrozoo)
    on top of the 9 Phase 1b params. These control total plankton availability to
    all fish and are the primary lever for reducing planktivore carrying capacity.

    R18 default is 0.8 for all LTL; realistic range 0.1–0.8.
    13 params total: 4 larval + 4 adult mortality + 1 cod ingestion + 4 LTL access.
    """
    keys, bounds, x0 = get_phase1b_params()

    # LTL accessibility: sp8=Diatoms, sp10=Microzoo, sp11=Mesozoo, sp12=Macrozoo
    # Skip sp9=Dinoflagellates (low accessibility already) and sp13=Benthos (not planktivore prey)
    ltl_species = [
        (8, "Diatoms"),
        (10, "Microzooplankton"),
        (11, "Mesozooplankton"),
        (12, "Macrozooplankton"),
    ]
    for sp_idx, _name in ltl_species:
        keys.append(f"species.accessibility2fish.sp{sp_idx}")
        bounds.append((np.log10(0.1), np.log10(0.8)))  # 0.1 to 0.8
        x0.append(np.log10(0.8))  # start at R18 default

    return keys, bounds, x0


def get_phase1e_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 1e: Phase 1b + sprat fishing rate (10 params, LTL pre-fixed at 0.3).

    Sprat dominates the objective at 11x target even after LTL=0.4 fix.
    Adding sprat fishing as a direct biomass lever, and lowering LTL to 0.3
    (diagnostic showed sprat baseline drops from 4.2x to 3.2x at 0.3).

    10 params: 4 larval + 4 adult mortality + 1 cod ingestion + 1 sprat fishing.
    """
    keys, bounds, x0 = get_phase1b_params()

    # Sprat fishing rate: R18 = 0.25, allow wide range
    keys.append("fisheries.rate.base.fsh2")
    bounds.append((np.log10(0.05), np.log10(2.0)))  # 0.05 to 2.0
    x0.append(np.log10(0.25))  # start at R18 default

    return keys, bounds, x0


def get_phase1f_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 1f: Phase 1b params with literature-adjusted accessibility matrix + LTL=0.4.

    Literature review (Lankov et al. 2010, Möllmann et al. 1998) showed herring and
    sprat have MORE different diets than the R18 matrix assumed (both had 0.8 for all
    plankton). Herring targets larger zooplankton, sprat smaller items.

    The predation-accessibility.csv has been updated with differentiated values:
      Herring: Diatoms 0.8→0.5, Microzoo 0.8→0.5, Mesozoo 0.8 (kept), Macrozoo 0.5→0.7
      Sprat:   Diatoms 0.8 (kept), Microzoo 0.8 (kept), Mesozoo 0.8→0.5, Macrozoo 0.4→0.3

    Diagnostic showed this halved planktivore overshoot at baseline:
      Herring: 10x→5.0x, Sprat: 10x→4.0x (with LTL=0.4).

    9 params: 4 larval + 4 adult mortality + 1 cod ingestion (same as Phase 1b).
    """
    return get_phase1b_params()


def get_phase1g_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 1g: Literature-validated baselines + dynamic accessibility.

    Baseline corrections applied (literature-validated):
      - Species-specific ingestion rates (herring 6.0, sprat 7.0, stickleback 5.0)
      - Stickleback larval mortality 0.4→3.5 (was 20-39x too low)
      - Flounder Linf 50→42 cm (FishBase Baltic range)
      - Herring Linf 30→27 cm (Central Baltic specific)
      - Herring F 0.06→0.15, Sprat F 0.25→0.32 (closer to ICES)
      - Stickleback fecundity 800→400 eggs/g

    Dynamic accessibility enabled (prey density-dependent scaling).

    Free params: 4 larval + 4 adult mortality + cod ingestion (same as Phase 1b).
    The literature corrections change the starting point, not the free parameters.
    """
    # Same free params as 1b, but starting from corrected baselines
    problem_species = [1, 2, 3, 7]  # herring, sprat, flounder, stickleback

    keys = []
    bounds = []
    x0 = []

    # Updated R18 larval mortality baselines (stickleback corrected from 0.4 to 3.5)
    r18_larval = {1: 8.0, 2: 9.0, 3: 12.0, 7: 3.5}
    r18_adult = {1: 0.001, 2: 0.05, 3: 0.05, 7: 0.001}

    for sp_idx in problem_species:
        keys.append(f"mortality.additional.larva.rate.sp{sp_idx}")
        bounds.append((-0.5, 2.0))  # 0.3 to 100
        x0.append(np.log10(max(r18_larval[sp_idx], 0.1)))

    for sp_idx in problem_species:
        keys.append(f"mortality.additional.rate.sp{sp_idx}")
        bounds.append((-2.5, 0.5))  # 0.003 to 3.16
        x0.append(np.log10(max(r18_adult[sp_idx], 0.003)))

    # Cod ingestion rate: parasite-adjusted (Ryberg et al. 2020)
    keys.append("predation.ingestion.rate.max.sp0")
    bounds.append((np.log10(1.5), np.log10(4.0)))  # 1.5 to 4.0
    x0.append(np.log10(3.5))  # start at R18 default

    return keys, bounds, x0


def get_phase2_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 2: fishing rates refinement (8 params, log10 space).

    Run after Phase 1 to fine-tune with fishing pressure.
    """
    keys = []
    bounds = []
    x0 = []

    r18_fishing = [0.08, 0.06, 0.25, 0.04, 0.03, 0.03, 0.02, 0.01]
    for i in range(N_SPECIES):
        keys.append(f"fisheries.rate.base.fsh{i}")
        bounds.append((-2.5, 0.0))
        x0.append(np.log10(max(r18_fishing[i], 0.003)))

    return keys, bounds, x0


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------
def run_calibration(
    phase: str,
    maxiter: int = 200,
    n_seeds: int = 1,
    n_years: int = 40,
    popsize: int = 15,
) -> dict:
    """Run differential evolution calibration for the specified phase."""
    from osmose.config.reader import OsmoseConfigReader

    print(f"=== Baltic Sea Calibration — Phase {phase} ===", flush=True)
    print(f"Config: {BALTIC_CONFIG}")
    print(f"Simulation years: {n_years}, DE maxiter: {maxiter}, popsize: {popsize}", flush=True)

    # Load base config
    reader = OsmoseConfigReader()
    base_config = reader.read(BALTIC_CONFIG)

    # Load targets
    targets = load_targets()
    print(f"Targets loaded for {len(targets)} species")
    for t in targets:
        print(f"  {t.species}: {t.target:,.0f} t (range {t.lower:,.0f}–{t.upper:,.0f})")

    # Get phase params
    if phase == "1":
        param_keys, bounds, x0 = get_phase1_params()
    elif phase == "1b":
        param_keys, bounds, x0 = get_phase1b_params()
    elif phase == "1c":
        param_keys, bounds, x0 = get_phase1c_params()
    elif phase == "1d":
        param_keys, bounds, x0 = get_phase1b_params()  # same 9 params as 1b
    elif phase == "1e":
        param_keys, bounds, x0 = get_phase1e_params()  # 1b + sprat fishing
    elif phase == "1f":
        param_keys, bounds, x0 = get_phase1f_params()  # 1b params, literature matrix
    elif phase == "1g":
        param_keys, bounds, x0 = get_phase1g_params()  # literature-validated baselines
    elif phase == "2":
        param_keys, bounds, x0 = get_phase2_params()
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Phase 1d: pre-fix LTL plankton accessibility at 0.4 (down from R18 0.8)
    if phase == "1d":
        for sp_idx in [8, 9, 10, 11, 12, 13]:
            base_config[f"species.accessibility2fish.sp{sp_idx}"] = "0.4"
        print("Phase 1d: LTL accessibility pre-fixed at 0.4 (structural correction)")

    # Phase 1e: pre-fix LTL accessibility at 0.3 (aggressive plankton reduction)
    # Diagnostic: sprat baseline drops from 4.2x (at 0.4) to 3.2x (at 0.3)
    if phase == "1e":
        for sp_idx in [8, 9, 10, 11, 12, 13]:
            base_config[f"species.accessibility2fish.sp{sp_idx}"] = "0.3"
        print("Phase 1e: LTL accessibility pre-fixed at 0.3 + sprat fishing free")

    # Phase 1f: literature-adjusted accessibility matrix (already in CSV) + LTL=0.4
    # Herring/sprat plankton access differentiated per Lankov et al. 2010 & Möllmann 1998
    if phase == "1f":
        for sp_idx in [8, 9, 10, 11, 12, 13]:
            base_config[f"species.accessibility2fish.sp{sp_idx}"] = "0.4"
        print("Phase 1f: Literature-adjusted matrix + LTL=0.4")

    # Phase 1g: literature-validated baselines + dynamic accessibility + LTL=0.4
    # All baseline corrections already applied to config files
    if phase == "1g":
        for sp_idx in [8, 9, 10, 11, 12, 13]:
            base_config[f"species.accessibility2fish.sp{sp_idx}"] = "0.4"
        base_config["predation.accessibility.dynamic.enabled"] = "true"
        base_config["predation.accessibility.dynamic.exponent"] = "1.0"
        base_config["predation.accessibility.dynamic.floor"] = "0.05"
        print("Phase 1g: Literature-validated baselines + dynamic accessibility + LTL=0.4")

    print(f"\nFree parameters ({len(param_keys)}):")
    for key, (lo, hi), x0_val in zip(param_keys, bounds, x0):
        config_key = key.lower()
        current = base_config.get(config_key, "?")
        print(f"  {key}: log10 bounds [{lo:.1f}, {hi:.1f}] (current: {current}, x0: {x0_val:.3f})")

    # Build objective
    objective = make_objective(
        base_config, targets, param_keys,
        n_years=n_years, seed=42, use_log_space=True,
    )

    # Initialize DE population: mixed strategy
    # - First member: exact R18 values
    # - 30% near R18 (±15% perturbation for local exploitation)
    # - 70% Latin Hypercube for global exploration
    n_params = len(param_keys)
    eff_popsize = max(popsize, 10 * n_params)  # at least 10× n_params
    rng = np.random.default_rng(42)
    x0_arr = np.array(x0)
    bounds_arr = np.array(bounds)
    widths = bounds_arr[:, 1] - bounds_arr[:, 0]

    n_r18 = max(1, int(0.3 * eff_popsize))  # 30% near R18
    n_lhs = eff_popsize - n_r18  # 70% LHS global

    init_pop = np.zeros((eff_popsize, n_params))

    # First member: exact R18
    init_pop[0] = x0_arr

    # R18 neighborhood (±15% of bound range)
    for i in range(1, n_r18):
        perturbation = rng.uniform(-0.15, 0.15, n_params) * widths
        candidate = x0_arr + perturbation
        init_pop[i] = np.clip(candidate, bounds_arr[:, 0], bounds_arr[:, 1])

    # Latin Hypercube Sampling for global coverage
    from scipy.stats.qmc import LatinHypercube
    lhs_sampler = LatinHypercube(d=n_params, seed=rng)
    lhs_samples = lhs_sampler.random(n=n_lhs)  # [0, 1]^d
    for i in range(n_lhs):
        init_pop[n_r18 + i] = bounds_arr[:, 0] + lhs_samples[i] * widths

    # Run DE with improved settings
    print(f"\nStarting differential evolution (eff_popsize={eff_popsize}, maxiter={maxiter})...")
    print(f"Init: {n_r18} near R18, {n_lhs} LHS global")
    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        init=init_pop,
        seed=42,
        tol=0.001,
        mutation=(0.5, 1.5),
        recombination=0.8,
        disp=True,
        polish=False,  # L-BFGS-B unreliable on noisy/discontinuous landscape
    )

    elapsed = time.time() - t0
    print(f"\nDE completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Best DE objective (single-seed): {result.fun:.6f}")

    # Multi-seed re-ranking of DE best solution
    # Single-seed optimization can overfit; validate on multiple seeds
    rerank_seeds = [42, 123, 7, 999, 2024]
    print(f"\n=== Multi-seed re-ranking ({len(rerank_seeds)} seeds) ===")
    best_overrides: dict[str, str] = {}
    for i, key in enumerate(param_keys):
        log_val = result.x[i]
        actual_val = 10.0 ** log_val
        best_overrides[key] = str(actual_val)

    rerank_objectives = []
    for rs in rerank_seeds:
        obj_fn = make_objective(
            base_config, targets, param_keys,
            n_years=n_years, seed=rs, use_log_space=True,
        )
        obj_val = obj_fn(result.x)
        rerank_objectives.append(obj_val)

    mean_obj = float(np.mean(rerank_objectives))
    std_obj = float(np.std(rerank_objectives))
    print(f"  Per-seed objectives: {[f'{v:.4f}' for v in rerank_objectives]}")
    print(f"  Mean: {mean_obj:.4f} ± {std_obj:.4f}")
    print(f"  (Single-seed was: {result.fun:.4f})")

    # Decode results
    optimized: dict[str, float] = {}
    print("\n=== Optimized Parameters ===")
    for i, key in enumerate(param_keys):
        log_val = result.x[i]
        actual_val = 10.0 ** log_val
        optimized[key] = actual_val
        print(f"  {key}: {actual_val:.6f} (log10: {log_val:.4f})")

    # Multi-seed validation
    if n_seeds > 1:
        print(f"\n=== Multi-seed validation ({n_seeds} seeds) ===")
        overrides = {k: str(v) for k, v in optimized.items()}
        all_stats = []
        for seed in range(n_seeds):
            stats = run_simulation(base_config, overrides, n_years=n_years, seed=seed)
            all_stats.append(stats)
            print(f"  Seed {seed}: " + ", ".join(
                f"{sp}={stats.get(f'{sp}_mean', 0):.0f}" for sp in SPECIES_NAMES
            ))

        # Mean across seeds
        print("\n  Mean across seeds:")
        for sp in SPECIES_NAMES:
            means = [s.get(f"{sp}_mean", 0) for s in all_stats]
            cvs = [s.get(f"{sp}_cv", 0) for s in all_stats]
            target = next((t.target for t in targets if t.species == sp), 0)
            ratio = np.mean(means) / target if target > 0 else float("inf")
            print(
                f"    {sp:15s}: {np.mean(means):>12,.0f} t "
                f"(target: {target:>10,.0f}, ratio: {ratio:.2f}, "
                f"mean CV: {np.mean(cvs):.3f})"
            )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"phase{phase.replace('/', '_')}_results.json"
    save_data = {
        "phase": phase,
        "objective_single_seed": float(result.fun),
        "objective_multiseed_mean": mean_obj,
        "objective_multiseed_std": std_obj,
        "objective_per_seed": [float(v) for v in rerank_objectives],
        "n_evaluations": result.nfev,
        "elapsed_seconds": elapsed,
        "parameters": {k: float(v) for k, v in optimized.items()},
        "log10_parameters": {k: float(result.x[i]) for i, k in enumerate(param_keys)},
        "bounds_log10": {k: list(b) for k, b in zip(param_keys, bounds)},
    }
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return save_data


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------
def validate_calibration(n_years: int = 50, n_seeds: int = 3) -> None:
    """Run a long validation simulation with calibrated parameters."""
    from osmose.config.reader import OsmoseConfigReader

    print(f"=== Baltic Sea Calibration — Validation ({n_years} years, {n_seeds} seeds) ===")

    reader = OsmoseConfigReader()
    base_config = reader.read(BALTIC_CONFIG)
    targets = load_targets()

    # Load best parameters from all phases
    overrides: dict[str, str] = {}
    for phase_num in [1, 2]:
        results_file = RESULTS_DIR / f"phase{phase_num}_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            for key, val in data["parameters"].items():
                overrides[key] = str(val)
            print(f"Loaded phase {phase_num} parameters ({len(data['parameters'])} params)")

    if not overrides:
        print("No calibration results found. Run calibration first.")
        return

    print(f"Total overrides: {len(overrides)}")
    for k, v in sorted(overrides.items()):
        print(f"  {k} = {v}")

    # Run validation
    print(f"\nRunning {n_seeds} validation simulations ({n_years} years each)...")
    all_stats = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}...")
        stats = run_simulation(base_config, overrides, n_years=n_years, seed=seed)
        all_stats.append(stats)

    # Summary
    target_dict = {t.species: t for t in targets}
    print("\n" + "=" * 90)
    print(f"{'Species':>15}  {'Mean Biomass':>14}  {'Target':>10}  "
          f"{'Ratio':>7}  {'CV':>6}  {'Trend':>7}  {'Status'}")
    print("-" * 90)

    all_pass = True
    for sp in SPECIES_NAMES:
        means = [s.get(f"{sp}_mean", 0) for s in all_stats]
        cvs = [s.get(f"{sp}_cv", 0) for s in all_stats]
        trends = [s.get(f"{sp}_trend", 0) for s in all_stats]

        mean_biomass = np.mean(means)
        mean_cv = np.mean(cvs)
        mean_trend = np.mean(trends)

        t = target_dict.get(sp)
        if t:
            ratio = mean_biomass / t.target
            in_range = t.lower <= mean_biomass <= t.upper
            stable = mean_cv < 0.3
            at_eq = mean_trend < 0.1
            status = "✅ PASS" if (in_range and stable and at_eq) else "❌ FAIL"
            if not (in_range and stable and at_eq):
                all_pass = False
            reasons = []
            if not in_range:
                reasons.append("OoR")
            if not stable:
                reasons.append("unstable")
            if not at_eq:
                reasons.append("trending")
            if reasons:
                status += f" ({','.join(reasons)})"
        else:
            ratio = float("nan")
            status = "? no target"

        print(
            f"{sp:>15}  {mean_biomass:>14,.0f}  {t.target if t else 0:>10,.0f}  "
            f"{ratio:>7.2f}  {mean_cv:>6.3f}  {mean_trend:>7.4f}  {status}"
        )

    print("=" * 90)
    print(f"Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILURES ❌'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baltic Sea OSMOSE calibration")
    parser.add_argument("--phase", type=str, default="1b",
                        help="Calibration phase: 1=all 16p, 1b=focused 8p, 2=fishing 8p")
    parser.add_argument("--maxiter", type=int, default=200, help="DE max iterations")
    parser.add_argument("--popsize", type=int, default=15, help="DE population size multiplier")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for validation")
    parser.add_argument("--years", type=int, default=40, help="Simulation years per eval")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    args = parser.parse_args()

    if args.validate:
        validate_calibration(n_years=50, n_seeds=args.seeds)
    else:
        run_calibration(
            phase=args.phase,
            maxiter=args.maxiter,
            n_seeds=args.seeds,
            n_years=args.years,
            popsize=args.popsize,
        )


if __name__ == "__main__":
    main()
