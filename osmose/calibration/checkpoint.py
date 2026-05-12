"""Calibration checkpoint module — atomic on-disk progress snapshots.

Read by the Shiny dashboard at 1 Hz; written by every optimizer
(DE / CMA-ES / surrogate-DE / NSGA-II) every N generations.

See docs/superpowers/specs/2026-05-12-calibration-dashboard-design.md for the
full contract and 14 invariants enforced in CalibrationCheckpoint.__post_init__.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

MAX_CHECKPOINT_BYTES: Final[int] = 1_048_576  # 1 MiB; real checkpoints are ~10 KB

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def default_results_dir() -> Path:
    """Baltic default — package-root-resolved data/baltic/calibration_results/.

    Callers may pass a different directory to write_checkpoint / read_checkpoint
    to support non-Baltic configurations.
    """
    return _PACKAGE_ROOT / "data" / "baltic" / "calibration_results"


# Single source-of-truth for the Baltic results directory. Both
# scripts/calibrate_baltic.py and ui/pages/calibration_handlers.py import this
# instead of redeclaring their own copy — keeps tmp_results_dir's monkeypatch
# to ONE target. See Task 8 fixture notes.
RESULTS_DIR: Final[Path] = default_results_dir()


_PHASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]*$")


@dataclass(frozen=True)
class CalibrationCheckpoint:
    """Atomic snapshot of a calibration run, written every N generations.

    See spec §5 for the full contract and the 14 invariants enforced below.
    The class is JSON-serialised by write_checkpoint and reconstructed by
    read_checkpoint (which converts invariant ValueErrors to kind='corrupt').
    """

    optimizer: Literal["de", "cmaes", "surrogate-de", "nsga2"]
    phase: str
    generation: int
    generation_budget: int | None
    best_fun: float
    per_species_residuals: tuple[float, ...] | None
    per_species_sim_biomass: tuple[float, ...] | None
    species_labels: tuple[str, ...] | None
    best_x_log10: tuple[float, ...]
    best_parameters: dict[str, float]
    param_keys: tuple[str, ...]
    bounds_log10: dict[str, tuple[float, float]]
    gens_since_improvement: int
    elapsed_seconds: float
    timestamp_iso: str
    banded_targets: dict[str, tuple[float, float]] | None
    proxy_source: Literal["banded_loss", "objective_disabled", "not_implemented"]

    def __post_init__(self) -> None:
        # Inv 1: generation >= 0
        if self.generation < 0:
            raise ValueError(f"generation must be >= 0, got {self.generation}")
        # Inv 2: 0 <= gens_since_improvement <= generation
        if self.gens_since_improvement < 0 or self.gens_since_improvement > self.generation:
            raise ValueError(
                f"gens_since_improvement must be in [0, generation], "
                f"got {self.gens_since_improvement} (generation={self.generation})"
            )
        # Inv 3: elapsed_seconds >= 0
        if self.elapsed_seconds < 0:
            raise ValueError(f"elapsed_seconds must be >= 0, got {self.elapsed_seconds}")
        # Inv 4: best_fun finite
        if not math.isfinite(self.best_fun):
            raise ValueError(f"best_fun must be finite, got {self.best_fun}")
        # Inv 5: best_parameters.keys == param_keys
        if set(self.best_parameters.keys()) != set(self.param_keys):
            raise ValueError(
                f"best_parameters keys mismatch param_keys: "
                f"{set(self.best_parameters.keys()) ^ set(self.param_keys)}"
            )
        # Inv 6: bounds_log10.keys == param_keys
        if set(self.bounds_log10.keys()) != set(self.param_keys):
            raise ValueError(
                f"bounds_log10 keys mismatch param_keys: "
                f"{set(self.bounds_log10.keys()) ^ set(self.param_keys)}"
            )
        # Inv 7: best_x_log10 parallel to param_keys
        if len(self.best_x_log10) != len(self.param_keys):
            raise ValueError(
                f"best_x_log10 length {len(self.best_x_log10)} != "
                f"param_keys length {len(self.param_keys)}"
            )
        # Inv 8: bounds_log10 values are (lo, hi) with lo <= hi
        for k, b in self.bounds_log10.items():
            if len(b) != 2 or b[0] > b[1]:
                raise ValueError(f"bounds_log10[{k!r}] = {b!r} invalid (need (lo, hi), lo<=hi)")
        # Inv 9: banded_targets values are (lo, hi) with 0 < lo <= hi
        if self.banded_targets is not None:
            for sp, b in self.banded_targets.items():
                if len(b) != 2 or b[0] > b[1] or b[0] <= 0:
                    raise ValueError(
                        f"banded_targets[{sp!r}] = {b!r} invalid "
                        "(need (lo, hi), 0<lo<=hi for magnitude factor)"
                    )
        # Inv 10: per_species_residuals parallel to species_labels, finite, >= 0
        if self.per_species_residuals is not None:
            if self.species_labels is None:
                raise ValueError("per_species_residuals without species_labels")
            if len(self.per_species_residuals) != len(self.species_labels):
                raise ValueError(
                    f"per_species_residuals length {len(self.per_species_residuals)} "
                    f"!= species_labels length {len(self.species_labels)}"
                )
            for i, r in enumerate(self.per_species_residuals):
                if not math.isfinite(r) or r < 0:
                    raise ValueError(
                        f"per_species_residuals[{i}] = {r} not finite or negative"
                    )
        # Inv 11: every species_label has a banded_targets entry
        if self.species_labels is not None and self.banded_targets is not None:
            missing = set(self.species_labels) - set(self.banded_targets.keys())
            if missing:
                raise ValueError(f"species_labels missing from banded_targets: {missing}")
        # Inv 12: proxy_source == "banded_loss" iff per_species_residuals is not None
        if (self.proxy_source == "banded_loss") != (self.per_species_residuals is not None):
            raise ValueError(
                f"proxy_source={self.proxy_source!r} inconsistent with "
                f"per_species_residuals is{'' if self.per_species_residuals is not None else ' not'} None"
            )
        # Inv 13: per_species_sim_biomass iff per_species_residuals, parallel, finite, >= 0
        if (self.per_species_sim_biomass is None) != (self.per_species_residuals is None):
            raise ValueError(
                "per_species_sim_biomass must be set iff per_species_residuals is set"
            )
        if self.per_species_sim_biomass is not None:
            assert self.species_labels is not None  # guarded by Inv 10
            if len(self.per_species_sim_biomass) != len(self.species_labels):
                raise ValueError(
                    f"per_species_sim_biomass length {len(self.per_species_sim_biomass)} "
                    f"!= species_labels length {len(self.species_labels)}"
                )
            for i, b in enumerate(self.per_species_sim_biomass):
                if not math.isfinite(b) or b < 0:
                    raise ValueError(
                        f"per_species_sim_biomass[{i}] = {b} not finite or negative"
                    )
        # Inv 14: phase regex + length + no `..`
        if not (1 <= len(self.phase) <= 64):
            raise ValueError(f"phase length must be 1..64, got {len(self.phase)}")
        if ".." in self.phase:
            raise ValueError(f"phase must not contain '..': {self.phase!r}")
        if not _PHASE_RE.match(self.phase):
            raise ValueError(f"phase {self.phase!r} fails regex {_PHASE_RE.pattern}")
