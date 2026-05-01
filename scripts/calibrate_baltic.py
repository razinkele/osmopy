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
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
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
    "perch", "pikeperch", "smelt", "stickleback",
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
class _ObjectiveWrapper:
    """Picklable wrapper for objective function to support multiprocessing.

    scipy.optimize.differential_evolution with workers>1 requires the objective
    function to be picklable, so we wrap it in a class with __call__.
    The closure pattern (returning a nested function) is not picklable by default.
    """

    def __init__(
        self,
        base_config: dict[str, str],
        targets: list[BiomassTarget],
        param_keys: list[str],
        n_years: int = 40,
        seed: int = 42,
        use_log_space: bool = True,
        w_stability: float = 5.0,
        w_worst: float = 0.5,
    ):
        self.base_config = base_config
        self.targets = targets
        self.target_dict = {t.species: t for t in targets}
        self.param_keys = param_keys
        self.n_years = n_years
        self.seed = seed
        self.use_log_space = use_log_space
        self.w_stability = w_stability
        self.w_worst = w_worst

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate objective function at point x."""
        # Map parameter vector to config overrides
        overrides: dict[str, str] = {}
        for i, key in enumerate(self.param_keys):
            if self.use_log_space:
                val = 10.0 ** x[i]
            else:
                val = x[i]
            overrides[key] = str(val)

        # Run simulation
        stats = run_simulation(
            self.base_config, overrides, n_years=self.n_years, seed=self.seed
        )
        if not stats:
            return 1e6  # Failed run

        # Compute objective
        total_error = 0.0
        worst_error = 0.0
        for sp in SPECIES_NAMES:
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in stats or sp not in self.target_dict:
                total_error += 100.0
                worst_error = max(worst_error, 100.0)
                continue

            sim_biomass = stats[mean_key]
            target = self.target_dict[sp]

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
                total_error += self.w_stability * target.weight * (cv - 0.2) ** 2

            # Trend penalty: penalize non-equilibrium
            trend = stats.get(trend_key, 0.0)
            if trend > 0.05:
                total_error += self.w_stability * target.weight * (trend - 0.05) ** 2

        # Add worst-species term to prevent hiding one bad species
        total_error += self.w_worst * worst_error

        return total_error


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

    Returns a picklable _ObjectiveWrapper instance to support multiprocessing
    in scipy.optimize.differential_evolution.
    """
    return _ObjectiveWrapper(
        base_config=base_config,
        targets=targets,
        param_keys=param_keys,
        n_years=n_years,
        seed=seed,
        use_log_space=use_log_space,
        w_stability=w_stability,
        w_worst=w_worst,
    )


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
        (-1.0, 2.0),   # sp6 smelt
        (-1.0, 2.0),   # sp7 stickleback
    ]
    for i in range(N_SPECIES):
        keys.append(f"mortality.additional.larva.rate.sp{i}")
        bounds.append(larval_bounds[i])
        x0.append(np.log10(max(r18_larval[i], 0.1)))

    # Adult additional mortality: R18 values range 0.0-0.05
    # Bounds: log10(0.001)=-3.0 to log10(2.0)=0.3 by default.
    # Widened to log10(5.0)=0.7 for species with documented seal/cormorant
    # predation pressure (Lundström 2010, Östman 2013, Heikinheimo 2021).
    # B-H stock-recruitment (v0.11.0) now provides density-dependent
    # cod recruitment, so DE no longer needs a hard mortality floor —
    # the linear-eggs compensation pathway is closed at the egg stage.
    r18_adult = [0.05, 0.001, 0.05, 0.05, 0.02, 0.03, 0.02, 0.001]
    adult_bounds = [
        (-3.0, 0.7),   # sp0 cod — seal predation
        (-3.0, 0.7),   # sp1 herring — heavy seal predation (16-19% per Gårdmark 2012)
        (-3.0, 0.7),   # sp2 sprat — seal predation + cormorant on juveniles
        (-3.0, 0.7),   # sp3 flounder — seal predation + some cormorant
        (-3.0, 0.7),   # sp4 perch — cormorant predation (4-10% per Heikinheimo 2021)
        (-3.0, 0.7),   # sp5 pikeperch — cormorant predation (4-23% per Heikinheimo 2016)
        (-3.0, 0.3),   # sp6 smelt — no documented top predator in model, keep default
        (-3.0, 0.3),   # sp7 stickleback — boom-bust, not predator-limited
    ]
    for i in range(N_SPECIES):
        keys.append(f"mortality.additional.rate.sp{i}")
        bounds.append(adult_bounds[i])
        # Defensive: clamp x0 into its bound so DE doesn't start from an
        # infeasible point if a per-species floor or ceiling is later raised.
        log_x0 = np.log10(max(r18_adult[i], 0.001))
        log_x0 = max(adult_bounds[i][0], min(adult_bounds[i][1], log_x0))
        x0.append(log_x0)

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
    # Per-species upper bounds: widen for flounder (sp3) and pikeperch (sp5) because
    # the 2026-04-24 phase 2 calibration had fsh3 pinned at the log10=0.0 ceiling —
    # DE wanted more fishing pressure. These two species have no natural-predator
    # control in the 8-species model, so fishing is the only lever until background
    # predators are added.
    fishing_upper = [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    for i in range(N_SPECIES):
        keys.append(f"fisheries.rate.base.fsh{i}")
        bounds.append((-2.5, fishing_upper[i]))
        x0.append(np.log10(max(r18_fishing[i], 0.003)))

    return keys, bounds, x0


def get_recruitment_ssbhalf_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Beverton-Holt ssb_half DE parameters (3 species).

    Cod (sp0) is held fixed at 120 kt (Bpa per ICES cod.27.24-32) via the
    baltic_param-reproduction.csv default; only sp3/sp4/sp5 are tunable.

    Bounds in log10(tonnes):
    - sp3 flounder: 5–200 kt (lumped Baltic stock; ICES splits into 4 sub-stocks)
    - sp4 perch:    0.5–50 kt (Curonian Lagoon, no ICES assessment)
    - sp5 pikeperch: 0.5–50 kt (Curonian Lagoon, no ICES assessment)
    """
    keys = [
        "stock.recruitment.ssbhalf.sp3",
        "stock.recruitment.ssbhalf.sp4",
        "stock.recruitment.ssbhalf.sp5",
    ]
    bounds = [
        (3.7, 5.3),  # sp3 flounder: 5k–200k t
        (2.7, 4.7),  # sp4 perch:    500–50k t
        (2.7, 4.7),  # sp5 pikeperch: 500–50k t
    ]
    x0 = [
        np.log10(50000.0),  # sp3 flounder mid-range
        np.log10(10000.0),  # sp4 perch mid-range
        np.log10(10000.0),  # sp5 pikeperch mid-range
    ]
    return keys, bounds, x0


def get_phase12_params() -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Phase 12: joint phase 1 + phase 2 + B-H ssb_half (27 params).

    Concatenates all mortality + fishing + recruitment params for joint
    optimization. Captures predator-prey feedback that sequential phase1→phase2
    missed, plus density-dependent recruitment (v0.11.0).
    """
    keys1, bounds1, x01 = get_phase1_params()
    keys2, bounds2, x02 = get_phase2_params()
    keys3, bounds3, x03 = get_recruitment_ssbhalf_params()
    return keys1 + keys2 + keys3, bounds1 + bounds2 + bounds3, x01 + x02 + x03


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------
_OPTIMIZER_CHOICES = ("de", "cmaes", "surrogate-de")


def _make_checkpoint_callback(
    checkpoint_path: Path,
    every_n: int,
    param_keys: list[str],
    bounds: list[tuple[float, float]],
):
    """Build a scipy DE callback that snapshots the current best every N gens.

    Solves the "long DE run interrupted = total loss of best x" problem we hit
    on 2026-04-30 when a 31h run found f=2.499 but had no way to surface its
    best params on SIGTERM. With this callback, killing the calibration at
    any point still leaves a usable JSON snapshot on disk.

    Atomic write via tmp + rename — a kill mid-write leaves the prior snapshot
    intact rather than a partial file. Uses scipy 1.11+ callback signature
    where the argument is an OptimizeResult-like object exposing .x and .fun.
    """
    state = {"gen": 0}

    def callback(intermediate_result, *_args, **_kw):
        state["gen"] += 1
        if every_n <= 0 or state["gen"] % every_n != 0:
            return None
        try:
            best_x = np.asarray(intermediate_result.x, dtype=float)
            best_fun = float(intermediate_result.fun)
        except AttributeError:
            # Fallback if scipy passes the legacy (xk, convergence) signature
            return None
        snapshot = {
            "generation": state["gen"],
            "best_fun": best_fun,
            "best_x_log10": [float(v) for v in best_x],
            "best_parameters": {
                k: float(10.0 ** best_x[i])
                for i, k in enumerate(param_keys)
            },
            "bounds_log10": {k: list(b) for k, b in zip(param_keys, bounds)},
            "timestamp_iso": datetime.now().isoformat(),
        }
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        tmp_path.replace(checkpoint_path)
        return None

    return callback


def _dispatch_optimizer(
    optimizer: str,
    objective,
    bounds: list[tuple[float, float]],
    x0: list[float],
    init_pop: np.ndarray,
    *,
    maxiter: int,
    popsize: int,
    tol: float,
    workers: int,
    seed: int,
    de_callback=None,
) -> dict:
    """Dispatch to the chosen optimizer; return a normalised result dict.

    Result schema: ``{x, fun, nfev, success, message}``. Optimizer-specific
    extras (``history``, ``X_train``, etc.) pass through unchanged.

    DE consumes the pre-built ``init_pop`` (LHS + R18 neighbourhood).
    CMA-ES and surrogate-DE generate their own initial samples internally
    and only need ``x0`` — the ``init_pop`` is unused for those paths.

    ``de_callback``: optional scipy DE callback (typically built by
    ``_make_checkpoint_callback``). Only consumed by the DE branch; CMA-ES
    and surrogate-DE have separate iteration models and ignore it.
    """
    if optimizer == "de":
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            init=init_pop,
            seed=seed,
            tol=tol,
            mutation=(0.5, 1.5),
            recombination=0.8,
            disp=True,
            polish=False,  # L-BFGS-B unreliable on noisy landscape
            workers=workers,
            updating="deferred",  # required when workers > 1
            callback=de_callback,
        )
        return {
            "x": np.asarray(result.x, dtype=float),
            "fun": float(result.fun),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
        }
    if optimizer == "cmaes":
        from osmose.calibration.cmaes_runner import run_cmaes
        return run_cmaes(
            objective,
            bounds,
            x0=x0,
            sigma0=0.3,
            popsize=popsize if popsize > 0 else None,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            workers=workers,
            verbose=True,
        )
    if optimizer == "surrogate-de":
        from osmose.calibration.surrogate_de import surrogate_assisted_de
        n_dim = len(bounds)
        result = surrogate_assisted_de(
            objective,
            bounds,
            x0=x0,
            n_initial=max(20, 5 * n_dim),
            n_iterations=6,
            n_topk=30,
            workers=workers,
            seed=seed,
            verbose=True,
        )
        # Add scipy-style fields for downstream compatibility. surrogate-DE
        # has no convergence concept (fixed-iteration loop), so success is
        # whether the final training set is non-empty.
        result["success"] = bool(len(result.get("y_train", [])) > 0)
        result["message"] = (
            f"surrogate-DE: {len(result.get('history', [])) - 1} refinement iterations "
            f"completed; nfev={result['nfev']}"
        )
        return result
    raise ValueError(
        f"unknown optimizer: {optimizer!r}; choices are {_OPTIMIZER_CHOICES}"
    )


def apply_warm_start(
    warm_start_path: Path,
    param_keys: list[str],
    x0: list[float],
    skip_keys: set[str],
) -> tuple[list[float], list[str], list[str]]:
    """Override x0 entries from a prior calibration's log10_parameters.

    Returns (new_x0, applied_keys, skipped_keys). Keys in `skip_keys` are
    explicitly excluded (e.g. when bounds changed for that param). Keys absent
    from the JSON keep their computed x0 (e.g. ssb_half params not in a
    pre-B-H result).
    """
    with open(warm_start_path) as f:
        data = json.load(f)
    log_params = data.get("log10_parameters", {})
    new_x0 = list(x0)
    applied: list[str] = []
    skipped: list[str] = []
    for i, key in enumerate(param_keys):
        if key in skip_keys:
            skipped.append(key)
            continue
        if key in log_params:
            new_x0[i] = float(log_params[key])
            applied.append(key)
    return new_x0, applied, skipped


def run_calibration(
    phase: str,
    maxiter: int = 200,
    n_seeds: int = 1,
    n_years: int = 40,
    popsize: int = 15,
    popsize_mult: int = 10,
    tol: float = 0.005,
    warm_start_path: Path | None = None,
    skip_warm_start_keys: list[str] | None = None,
    optimizer: str = "de",
    checkpoint_every: int = 5,
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
    elif phase == "12":
        param_keys, bounds, x0 = get_phase12_params()
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

    # Phase 2: inherit calibrated mortality params from phase 1 as fixed base_config
    # overrides. Without this, phase 2 would run with R18 defaults, which would reintroduce
    # the smelt/stickleback extinction risk phase 1 just eliminated.
    if phase == "2":
        p1_file = RESULTS_DIR / "phase1_results.json"
        if p1_file.exists():
            with open(p1_file) as f:
                p1_data = json.load(f)
            for key, val in p1_data.get("parameters", {}).items():
                base_config[key.lower()] = str(val)
            print(f"Phase 2: inherited {len(p1_data.get('parameters', {}))} "
                  f"calibrated params from phase1_results.json")
        else:
            print("Phase 2 WARNING: no phase1_results.json found — running on R18 defaults "
                  "(risks re-introducing smelt/stickleback extinction).")

    # Phase 12: joint optimization. No inheritance needed; both phase 1 + 2 params are free.
    # This captures predator-prey feedback that sequential phase1→phase2 missed.
    if phase == "12":
        print("Phase 12: joint optimization of 27 params (16 mortality + 8 fishing "
              "+ 3 B-H ssb_half). Captures trophic cascade feedback + density-dependent "
              "recruitment.")

    # Apply warm-start (Tier B1): override x0 entries from a prior result JSON.
    # Keys absent from the JSON keep their computed default (e.g. new B-H
    # ssb_half params won't be in a pre-v0.11.0 result). Keys in skip set are
    # excluded — useful when bounds changed (e.g. cod sp0 after dropping the
    # cod-floor: old optimum sat at log10≈0.57 against a (-0.523, 0.7) bound,
    # which biases the new (-3.0, 0.7) search incorrectly post-B-H).
    if warm_start_path is not None:
        skip_set = set(skip_warm_start_keys or [])
        x0, applied, skipped = apply_warm_start(
            warm_start_path, param_keys, x0, skip_set
        )
        print(f"\nWarm-start: loaded {len(applied)} params from {warm_start_path}")
        if applied:
            print(f"  Applied: {', '.join(applied)}")
        if skipped:
            print(f"  Skipped (explicit): {', '.join(skipped)}")
        not_in_json = [
            k for k in param_keys
            if k not in applied and k not in skipped
        ]
        if not_in_json:
            print(f"  Not in JSON (kept default x0): {', '.join(not_in_json)}")

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
    eff_popsize = max(popsize, popsize_mult * n_params)
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

    # Run the chosen optimizer.
    # Workers cap default 8 (memory: each 50-y OSMOSE sim holds ~400 MB; 28
    # concurrent workers on a 28-thread box exhausted 32 GB and thrashed
    # 2026-04-24). 8 × 400 MB = 3.2 GB — comfortable. Override with
    # OSMOSE_DE_WORKERS env var.
    workers = int(os.environ.get("OSMOSE_DE_WORKERS", "8"))
    print(f"\nStarting optimizer={optimizer!r} (eff_popsize={eff_popsize}, "
          f"maxiter={maxiter}, workers={workers})...")
    print(f"Init: {n_r18} near R18, {n_lhs} LHS global "
          f"(consumed by DE; CMA-ES and surrogate-DE generate their own)")

    # Build the checkpoint callback for DE so a long run is interruptible
    # without losing the best-known x. Only meaningful for the DE path —
    # CMA-ES and surrogate-DE have their own iteration models.
    de_callback = None
    if optimizer == "de" and checkpoint_every > 0:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = RESULTS_DIR / f"phase{phase.replace('/', '_')}_checkpoint.json"
        de_callback = _make_checkpoint_callback(
            checkpoint_path, checkpoint_every, param_keys, bounds,
        )
        print(f"Checkpointing best-x every {checkpoint_every} generations to "
              f"{checkpoint_path}")

    t0 = time.time()

    result = _dispatch_optimizer(
        optimizer,
        objective,
        bounds,
        x0,
        init_pop,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        workers=workers,
        seed=42,
        de_callback=de_callback,
    )

    elapsed = time.time() - t0
    print(f"\n{optimizer} completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Function evaluations: {result['nfev']}")
    print(f"Best objective (single-seed): {result['fun']:.6f}")

    # Multi-seed re-ranking of best solution
    # Single-seed optimization can overfit; validate on multiple seeds
    rerank_seeds = [42, 123, 7, 999, 2024]
    print(f"\n=== Multi-seed re-ranking ({len(rerank_seeds)} seeds) ===")
    best_x = np.asarray(result["x"], dtype=float)
    best_overrides: dict[str, str] = {}
    for i, key in enumerate(param_keys):
        log_val = best_x[i]
        actual_val = 10.0 ** log_val
        best_overrides[key] = str(actual_val)

    rerank_objectives = []
    for rs in rerank_seeds:
        obj_fn = make_objective(
            base_config, targets, param_keys,
            n_years=n_years, seed=rs, use_log_space=True,
        )
        obj_val = obj_fn(best_x)
        rerank_objectives.append(obj_val)

    mean_obj = float(np.mean(rerank_objectives))
    std_obj = float(np.std(rerank_objectives))
    print(f"  Per-seed objectives: {[f'{v:.4f}' for v in rerank_objectives]}")
    print(f"  Mean: {mean_obj:.4f} ± {std_obj:.4f}")
    print(f"  (Single-seed was: {result['fun']:.4f})")

    # Decode results
    optimized: dict[str, float] = {}
    print("\n=== Optimized Parameters ===")
    for i, key in enumerate(param_keys):
        log_val = best_x[i]
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
        "optimizer": optimizer,
        "objective_single_seed": float(result["fun"]),
        "objective_multiseed_mean": mean_obj,
        "objective_multiseed_std": std_obj,
        "objective_per_seed": [float(v) for v in rerank_objectives],
        "n_evaluations": int(result["nfev"]),
        "elapsed_seconds": elapsed,
        "parameters": {k: float(v) for k, v in optimized.items()},
        "log10_parameters": {k: float(best_x[i]) for i, k in enumerate(param_keys)},
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
    parser.add_argument("--popsize", type=int, default=15, help="DE absolute population size floor")
    parser.add_argument("--popsize-mult", type=int, default=10,
                        help="DE population size multiplier of n_params (default 10)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for validation")
    parser.add_argument("--years", type=int, default=40, help="Simulation years per eval")
    parser.add_argument("--tol", type=float, default=0.005,
                        help="DE convergence tolerance (default 0.005; scipy default 0.01; "
                             "pre-2026-04-29 hardcoded 0.001 — Tier A3 speedup)")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to prior calibration JSON to warm-start x0 from (Tier B1)")
    parser.add_argument("--skip-warm-start-keys", type=str, default="",
                        help="Comma-separated param keys to exclude from warm-start "
                             "(e.g. when their bounds changed)")
    parser.add_argument("--optimizer", type=str, default="de",
                        choices=list(_OPTIMIZER_CHOICES),
                        help="Optimization algorithm. 'de' (default): scipy "
                             "differential_evolution, broad-search workhorse. 'cmaes': "
                             "CMA-ES via the cma package, 2-3× fewer evals on smooth "
                             "continuous landscapes (Tier C2). 'surrogate-de': GP-assisted "
                             "DE — trains a surrogate on real evals and runs DE on the "
                             "predicted objective; 5-10× fewer real evals when the "
                             "surrogate fits well (Tier C1).")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="DE only: snapshot best-x to a JSON file every N "
                             "generations. Lets you SIGTERM a long DE run and still "
                             "recover the best known parameters. Set to 0 to disable.")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    args = parser.parse_args()

    if args.validate:
        validate_calibration(n_years=50, n_seeds=args.seeds)
    else:
        skip_keys = [k.strip() for k in args.skip_warm_start_keys.split(",") if k.strip()]
        run_calibration(
            phase=args.phase,
            maxiter=args.maxiter,
            n_seeds=args.seeds,
            n_years=args.years,
            popsize=args.popsize,
            popsize_mult=args.popsize_mult,
            tol=args.tol,
            warm_start_path=Path(args.warm_start) if args.warm_start else None,
            skip_warm_start_keys=skip_keys,
            optimizer=args.optimizer,
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
