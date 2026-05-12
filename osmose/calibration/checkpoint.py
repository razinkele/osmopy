"""Calibration checkpoint module — atomic on-disk progress snapshots.

Read by the Shiny dashboard at 1 Hz; written by every optimizer
(DE / CMA-ES / surrogate-DE / NSGA-II) every N generations.

See docs/superpowers/specs/2026-05-12-calibration-dashboard-design.md for the
full contract and 14 invariants enforced in CalibrationCheckpoint.__post_init__.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

MAX_CHECKPOINT_BYTES: Final[int] = 1_048_576  # 1 MiB; real checkpoints are ~10 KB

_PARTIAL_WRITE_WINDOW_S: Final[float] = 3.0

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
        # Inv 5a: param_keys must be unique (set-equality above would
        # otherwise silently collapse duplicates)
        if len(self.param_keys) != len(set(self.param_keys)):
            raise ValueError(
                f"param_keys must be unique, got duplicates: {self.param_keys}"
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
        # Inv 12a: banded_loss requires banded_targets (the proxy table
        # renderer indexes banded_targets[species] for the magnitude factor)
        if self.proxy_source == "banded_loss" and self.banded_targets is None:
            raise ValueError(
                "proxy_source='banded_loss' requires banded_targets is not None"
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


@dataclass(frozen=True)
class CheckpointReadResult:
    """Discriminated read result.

    Lets the UI distinguish transient partial writes from persistent
    corruption from "no active run", without ever raising into the
    reactive runtime.
    """

    kind: Literal["ok", "no_run", "partial", "corrupt"]
    checkpoint: CalibrationCheckpoint | None
    error_summary: str | None

    def __post_init__(self) -> None:
        if self.kind == "ok" and self.checkpoint is None:
            raise ValueError(
                "CheckpointReadResult(kind='ok') requires non-None checkpoint; "
                "use kind='no_run' for the empty sentinel"
            )
        if self.kind != "ok" and self.checkpoint is not None:
            raise ValueError(
                f"CheckpointReadResult(kind={self.kind!r}) must have checkpoint=None"
            )


def _coerce_serialisable(value):
    """Coerce numpy scalars/arrays and tuples to plain Python types.

    Raises TypeError on unconvertible types; raises ValueError on non-finite floats.
    """
    if isinstance(value, dict):
        return {str(k): _coerce_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_serialisable(v) for v in value]
    if hasattr(value, "tolist"):  # numpy array / scalar
        value = value.tolist()
        if isinstance(value, (list, tuple)):
            return _coerce_serialisable(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float not serialisable: {value!r}")
        return float(value)
    raise TypeError(f"unserialisable value of type {type(value).__name__}: {value!r}")


def write_checkpoint(path: Path, ckpt: CalibrationCheckpoint) -> None:
    """Atomic write: serialise to a .tmp file then os.replace into place.

    Coerces numpy scalars/arrays to plain Python types at the boundary so DE/CMA-ES
    callers don't have to. Uses json.dump(allow_nan=False) as defence in depth
    against NaN that slipped past __post_init__.

    Raises (OSError, TypeError, ValueError) on failure; callers wrap in their own
    layered exception handler.
    """
    payload = _coerce_serialisable({
        "optimizer": ckpt.optimizer,
        "phase": ckpt.phase,
        "generation": ckpt.generation,
        "generation_budget": ckpt.generation_budget,
        "best_fun": ckpt.best_fun,
        "per_species_residuals": (
            list(ckpt.per_species_residuals)
            if ckpt.per_species_residuals is not None
            else None
        ),
        "per_species_sim_biomass": (
            list(ckpt.per_species_sim_biomass)
            if ckpt.per_species_sim_biomass is not None
            else None
        ),
        "species_labels": (
            list(ckpt.species_labels) if ckpt.species_labels is not None else None
        ),
        "best_x_log10": list(ckpt.best_x_log10),
        "best_parameters": dict(ckpt.best_parameters),
        "param_keys": list(ckpt.param_keys),
        "bounds_log10": {k: list(v) for k, v in ckpt.bounds_log10.items()},
        "gens_since_improvement": ckpt.gens_since_improvement,
        "elapsed_seconds": ckpt.elapsed_seconds,
        "timestamp_iso": ckpt.timestamp_iso,
        "banded_targets": (
            {k: list(v) for k, v in ckpt.banded_targets.items()}
            if ckpt.banded_targets is not None
            else None
        ),
        "proxy_source": ckpt.proxy_source,
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def read_checkpoint(path: Path) -> CheckpointReadResult:
    """Read and validate a checkpoint file. Never raises.

    See spec §5 for the four-kind contract:
      - 'ok'      : file present, JSON valid, all 14 invariants pass
      - 'no_run'  : file does not exist (vanished between glob and read)
      - 'partial' : decode/invariant error AND mtime within partial-write window
      - 'corrupt' : decode/invariant error AND older, OR size > MAX_CHECKPOINT_BYTES
    """
    path_str = str(path)
    try:
        st = path.stat()
    except (FileNotFoundError, PermissionError):
        return CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    except OSError as e:
        return CheckpointReadResult(
            kind="corrupt", checkpoint=None,
            error_summary=(
                f"stat failed for {path_str}: {e.__class__.__name__}: {e}. "
                "Recovery: delete the file or check filesystem mount/perms."
            ),
        )
    if st.st_size > MAX_CHECKPOINT_BYTES:
        return CheckpointReadResult(
            kind="corrupt", checkpoint=None,
            error_summary=(
                f"file {path_str} exceeds MAX_CHECKPOINT_BYTES "
                f"(size={st.st_size}, limit={MAX_CHECKPOINT_BYTES}). "
                "Recovery: delete the file — calibration will resume at next checkpoint."
            ),
        )

    age = time.time() - st.st_mtime
    is_recent = age < _PARTIAL_WRITE_WINDOW_S

    try:
        text = path.read_bytes().decode("utf-8")
        data = json.loads(text)
    except FileNotFoundError:
        return CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        kind = "partial" if is_recent else "corrupt"
        return CheckpointReadResult(
            kind=kind, checkpoint=None,
            error_summary=(
                f"decode failed for {path_str}: {e.__class__.__name__}: {e}. "
                "Recovery: if persistent, delete the file."
            ),
        )

    try:
        ckpt = CalibrationCheckpoint(
            optimizer=data["optimizer"],
            phase=data["phase"],
            generation=int(data["generation"]),
            generation_budget=data.get("generation_budget"),
            best_fun=float(data["best_fun"]),
            per_species_residuals=(
                tuple(data["per_species_residuals"])
                if data.get("per_species_residuals") is not None else None
            ),
            per_species_sim_biomass=(
                tuple(data["per_species_sim_biomass"])
                if data.get("per_species_sim_biomass") is not None else None
            ),
            species_labels=(
                tuple(data["species_labels"])
                if data.get("species_labels") is not None else None
            ),
            best_x_log10=tuple(data["best_x_log10"]),
            best_parameters=dict(data["best_parameters"]),
            param_keys=tuple(data["param_keys"]),
            bounds_log10={k: tuple(v) for k, v in data["bounds_log10"].items()},
            gens_since_improvement=int(data["gens_since_improvement"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
            timestamp_iso=data["timestamp_iso"],
            banded_targets=(
                {k: tuple(v) for k, v in data["banded_targets"].items()}
                if data.get("banded_targets") is not None else None
            ),
            proxy_source=data["proxy_source"],
        )
    except (KeyError, TypeError, ValueError) as e:
        # JSON decoded successfully, so the file is not a partial write —
        # it's either a frame from an older incompatible version or a real
        # invariant violation. Either way it's corrupt regardless of mtime.
        return CheckpointReadResult(
            kind="corrupt", checkpoint=None,
            error_summary=(
                f"invariant violation for {path_str}: {e.__class__.__name__}: {e}. "
                "Recovery: delete the file to discard the bad frame; the next "
                "successful generation re-creates it."
            ),
        )
    return CheckpointReadResult(kind="ok", checkpoint=ckpt, error_summary=None)


def is_live(path: Path, max_age_s: float = 60.0, now: float | None = None) -> bool:
    """True iff (now - max_age_s) < mtime <= now.

    Strict on the lower bound, inclusive on the upper. Future-mtime files
    (NTP rewind, clock jump) return False.
    """
    try:
        mtime = path.stat().st_mtime
    except (FileNotFoundError, PermissionError):
        return False
    t = time.time() if now is None else now
    return (t - max_age_s) < mtime <= t


def probe_writable(results_dir: Path) -> None:
    """Write a temporary probe file; raise OSError on permission/missing-dir failure.

    Uses NamedTemporaryFile(delete=True) so no sentinel persists after return.
    """
    with tempfile.NamedTemporaryFile(
        dir=results_dir, delete=True, prefix=".probe_", suffix=".tmp"
    ):
        pass


def liveness_state(age_seconds: float) -> Literal["live", "stalled", "idle"]:
    """Three-state liveness classification per spec §7."""
    if age_seconds < 0:
        return "idle"
    if age_seconds <= 60.0:
        return "live"
    if age_seconds <= 300.0:
        return "stalled"
    return "idle"
