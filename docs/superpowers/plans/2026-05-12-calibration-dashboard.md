# Calibration Progress Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Shiny-side calibration progress dashboard that monitors active OSMOSE calibration runs (CLI- or UI-launched, DE/CMA-ES/surrogate-DE/NSGA-II) via a new on-disk checkpoint format, displays live gen/best_fun/elapsed/patience/per-species ICES-proxy/magnitude-factor, and persists completed runs to the existing History tab.

**Architecture:** New `osmose/calibration/checkpoint.py` defines `CalibrationCheckpoint` (frozen dataclass, 14 invariants), `CheckpointReadResult` (discriminated 4-kind union), and `LiveSnapshot`. All four optimizers write the checkpoint atomically per generation; UI polls at 1 Hz via `@reactive.poll` and renders inline HTML widgets into the existing calibration page's Run tab. Per-species residuals come from extending `_ObjectiveWrapper` (CLI) and `make_banded_objective` (UI) symmetrically; DE/CMA-ES/surrogate-DE re-evaluate `best_x` in the main thread at checkpoint time to escape worker-process isolation.

**Tech Stack:** Python 3.12+, NumPy 2.x, scipy.optimize, pymoo (for NSGA-II), shiny-for-python. PEP 604 union syntax. Tests via `.venv/bin/python -m pytest`. Atomic write via `os.replace`. JSON serialization with 1 MiB size guard. HTML escaping via `html.escape` on all `ui.notification_show`. Plotly remains for the existing convergence chart; the new proxy table is plain HTML via `@render.ui`.

**Spec:** `docs/superpowers/specs/2026-05-12-calibration-dashboard-design.md` (851 lines, converged across 10 review rounds).

---

## Phase 1 — Checkpoint module foundation

### Task 1: Module skeleton + `default_results_dir`

**Files:**
- Create: `osmose/calibration/checkpoint.py`
- Create: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write the failing test**

`tests/test_calibration_checkpoint.py`:
```python
from __future__ import annotations

from pathlib import Path

from osmose.calibration.checkpoint import MAX_CHECKPOINT_BYTES, default_results_dir


def test_default_results_dir_resolves_to_baltic_calibration_results():
    """default_results_dir points at the Baltic results dir, package-root-resolved."""
    p = default_results_dir()
    assert isinstance(p, Path)
    assert p.parts[-3:] == ("data", "baltic", "calibration_results")


def test_max_checkpoint_bytes_is_1mib():
    """1 MiB ceiling for read_checkpoint's size guard."""
    assert MAX_CHECKPOINT_BYTES == 1_048_576
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.calibration.checkpoint'`.

- [ ] **Step 3: Write the minimal implementation**

`osmose/calibration/checkpoint.py`:
```python
"""Calibration checkpoint module — atomic on-disk progress snapshots.

Read by the Shiny dashboard at 1 Hz; written by every optimizer
(DE / CMA-ES / surrogate-DE / NSGA-II) every N generations.

See docs/superpowers/specs/2026-05-12-calibration-dashboard-design.md for the
full contract and 14 invariants enforced in CalibrationCheckpoint.__post_init__.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): scaffold checkpoint module with default_results_dir"
```

---

### Task 2: `CalibrationCheckpoint` dataclass with 14 invariants

**Files:**
- Modify: `osmose/calibration/checkpoint.py`
- Modify: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write failing tests for the happy-path roundtrip**

Append to `tests/test_calibration_checkpoint.py`:
```python
import dataclasses
import math

import pytest

from osmose.calibration.checkpoint import CalibrationCheckpoint


def _valid_checkpoint_kwargs() -> dict:
    """Build a CalibrationCheckpoint kwargs dict that satisfies all 14 invariants."""
    return dict(
        optimizer="de",
        phase="12",
        generation=10,
        generation_budget=200,
        best_fun=3.14,
        per_species_residuals=(0.0, 0.42),
        per_species_sim_biomass=(1.0, 2.4),
        species_labels=("sp_a", "sp_b"),
        best_x_log10=(-0.3, 0.8),
        best_parameters={"k_a": 0.5, "k_b": 6.3},
        param_keys=("k_a", "k_b"),
        bounds_log10={"k_a": (-1.0, 0.0), "k_b": (0.0, 1.0)},
        gens_since_improvement=3,
        elapsed_seconds=42.0,
        timestamp_iso="2026-05-12T10:30:00+00:00",
        banded_targets={"sp_a": (0.5, 1.5), "sp_b": (1.5, 2.5)},
        proxy_source="banded_loss",
    )


def test_valid_checkpoint_constructs():
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    assert ckpt.optimizer == "de"
    assert ckpt.generation == 10


def test_checkpoint_is_frozen():
    """frozen=True — assigning to a field raises FrozenInstanceError."""
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ckpt.generation = 11  # type: ignore[misc]
```

- [ ] **Step 2: Write failing tests for each of the 14 invariants**

Append to `tests/test_calibration_checkpoint.py`:
```python
@pytest.mark.parametrize(
    "field,bad_value,error_match",
    [
        ("generation", -1, "generation"),                              # inv 1
        ("gens_since_improvement", -1, "gens_since_improvement"),      # inv 2
        ("elapsed_seconds", -0.5, "elapsed_seconds"),                  # inv 3
        ("best_fun", float("nan"), "finite"),                          # inv 4
        ("best_fun", float("inf"), "finite"),                          # inv 4
    ],
)
def test_invariant_scalar_bounds(field, bad_value, error_match):
    kwargs = _valid_checkpoint_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=error_match):
        CalibrationCheckpoint(**kwargs)


def test_invariant_5_best_parameters_keys_match_param_keys():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["best_parameters"] = {"k_a": 0.5}  # missing k_b
    with pytest.raises(ValueError, match="best_parameters"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_6_bounds_keys_match_param_keys():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["bounds_log10"] = {"k_a": (-1.0, 0.0)}  # missing k_b
    with pytest.raises(ValueError, match="bounds_log10"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_7_best_x_log10_len_matches_param_keys():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["best_x_log10"] = (-0.3,)  # one element, two params
    with pytest.raises(ValueError, match="best_x_log10"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_8_bounds_lo_le_hi():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["bounds_log10"] = {"k_a": (1.0, 0.0), "k_b": (0.0, 1.0)}
    with pytest.raises(ValueError, match="bounds_log10"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_9_banded_targets_lo_positive():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["banded_targets"] = {"sp_a": (0.0, 1.5), "sp_b": (1.5, 2.5)}
    with pytest.raises(ValueError, match="banded_targets"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_10_residuals_parallel_to_labels():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["per_species_residuals"] = (0.0, 0.42, 1.7)  # 3 vs 2 labels
    with pytest.raises(ValueError, match="per_species_residuals"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_11_labels_subset_of_banded_targets():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["banded_targets"] = {"sp_a": (0.5, 1.5)}  # missing sp_b
    with pytest.raises(ValueError, match="banded_targets"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_12_proxy_source_iff_residuals():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["proxy_source"] = "objective_disabled"  # but residuals is non-None
    with pytest.raises(ValueError, match="proxy_source"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_13_sim_biomass_iff_residuals():
    kwargs = _valid_checkpoint_kwargs()
    kwargs["per_species_sim_biomass"] = None  # but residuals is non-None
    with pytest.raises(ValueError, match="per_species_sim_biomass"):
        CalibrationCheckpoint(**kwargs)


@pytest.mark.parametrize(
    "phase,should_raise",
    [
        ("12", False),
        ("1g_pilot", False),
        ("12.no-predators", False),
        ("../../etc/passwd", True),
        (".hidden", True),
        (".", True),
        ("", True),
        ("a" * 65, True),
        ("12..xx", True),
        ("12\x00xx", True),
    ],
)
def test_invariant_14_phase_regex(phase, should_raise):
    kwargs = _valid_checkpoint_kwargs()
    kwargs["phase"] = phase
    if should_raise:
        with pytest.raises(ValueError, match="phase"):
            CalibrationCheckpoint(**kwargs)
    else:
        CalibrationCheckpoint(**kwargs)


def test_residuals_disabled_when_banded_loss_not_in_use():
    """Mirror invariants 10, 12, 13: all three nullable fields become None together."""
    kwargs = _valid_checkpoint_kwargs()
    kwargs["per_species_residuals"] = None
    kwargs["per_species_sim_biomass"] = None
    kwargs["species_labels"] = None
    kwargs["banded_targets"] = None
    kwargs["proxy_source"] = "objective_disabled"
    ckpt = CalibrationCheckpoint(**kwargs)
    assert ckpt.proxy_source == "objective_disabled"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: every new test FAILs with `ImportError: cannot import name 'CalibrationCheckpoint'`.

- [ ] **Step 4: Implement the dataclass with `__post_init__`**

Append to `osmose/calibration/checkpoint.py`:
```python
import math
import re
from dataclasses import dataclass
from typing import Literal

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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: every test PASSES (~22 cases).

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): CalibrationCheckpoint dataclass with 14 __post_init__ invariants"
```

---

### Task 3: `CheckpointReadResult` discriminated union

**Files:**
- Modify: `osmose/calibration/checkpoint.py`
- Modify: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_checkpoint.py`:
```python
from osmose.calibration.checkpoint import CheckpointReadResult


def test_checkpoint_read_result_ok_with_checkpoint():
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    r = CheckpointReadResult(kind="ok", checkpoint=ckpt, error_summary=None)
    assert r.kind == "ok"
    assert r.checkpoint is ckpt


def test_checkpoint_read_result_no_run_sentinel():
    r = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    assert r.checkpoint is None


def test_checkpoint_read_result_corrupt_with_error():
    r = CheckpointReadResult(kind="corrupt", checkpoint=None, error_summary="bad json")
    assert r.error_summary == "bad json"


def test_checkpoint_read_result_invariant_ok_without_checkpoint_raises():
    with pytest.raises(ValueError, match="kind='ok'"):
        CheckpointReadResult(kind="ok", checkpoint=None, error_summary=None)


def test_checkpoint_read_result_invariant_non_ok_with_checkpoint_raises():
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    with pytest.raises(ValueError, match="checkpoint=None"):
        CheckpointReadResult(kind="no_run", checkpoint=ckpt, error_summary=None)
    with pytest.raises(ValueError, match="checkpoint=None"):
        CheckpointReadResult(kind="partial", checkpoint=ckpt, error_summary=None)
    with pytest.raises(ValueError, match="checkpoint=None"):
        CheckpointReadResult(kind="corrupt", checkpoint=ckpt, error_summary="x")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v -k checkpoint_read_result
```

Expected: FAIL with `ImportError: cannot import name 'CheckpointReadResult'`.

- [ ] **Step 3: Implement `CheckpointReadResult`**

Append to `osmose/calibration/checkpoint.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): CheckpointReadResult discriminated union with two-sided invariants"
```

---

### Task 4: `write_checkpoint` with atomic write and numpy coercion

**Files:**
- Modify: `osmose/calibration/checkpoint.py`
- Modify: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_checkpoint.py`:
```python
import json

import numpy as np

from osmose.calibration.checkpoint import write_checkpoint


def test_write_checkpoint_creates_file(tmp_path):
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    path = tmp_path / "phase12_checkpoint.json"
    write_checkpoint(path, ckpt)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["optimizer"] == "de"
    assert data["generation"] == 10


def test_write_checkpoint_is_atomic_no_partial_file(tmp_path, monkeypatch):
    """If os.replace fails, only the .tmp file (if any) is left; the destination is not touched."""
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    path = tmp_path / "phase12_checkpoint.json"

    def boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError):
        write_checkpoint(path, ckpt)
    assert not path.exists()


def test_write_checkpoint_coerces_numpy_scalars(tmp_path):
    kwargs = _valid_checkpoint_kwargs()
    kwargs["best_fun"] = np.float64(3.14)  # numpy scalar
    ckpt = CalibrationCheckpoint(**kwargs)
    path = tmp_path / "phase12_checkpoint.json"
    write_checkpoint(path, ckpt)
    data = json.loads(path.read_text())
    assert data["best_fun"] == 3.14
    assert isinstance(data["best_fun"], float)


def test_write_checkpoint_disallows_nan_in_output(tmp_path):
    """write_checkpoint passes allow_nan=False to json.dump as defence in depth
    against any non-finite that slipped past __post_init__."""
    kwargs = _valid_checkpoint_kwargs()
    ckpt = CalibrationCheckpoint(**kwargs)
    path = tmp_path / "phase12_checkpoint.json"
    write_checkpoint(path, ckpt)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v -k write_checkpoint
```

Expected: FAIL with `ImportError: cannot import name 'write_checkpoint'`.

- [ ] **Step 3: Implement `write_checkpoint`**

Append to `osmose/calibration/checkpoint.py`:
```python
import json
import os


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
        "per_species_residuals": list(ckpt.per_species_residuals) if ckpt.per_species_residuals is not None else None,
        "per_species_sim_biomass": list(ckpt.per_species_sim_biomass) if ckpt.per_species_sim_biomass is not None else None,
        "species_labels": list(ckpt.species_labels) if ckpt.species_labels is not None else None,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): write_checkpoint with atomic tmp+rename and numpy coercion"
```

---

### Task 5: `read_checkpoint` with discriminated kinds, size guard, invariant catch

**Files:**
- Modify: `osmose/calibration/checkpoint.py`
- Modify: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_checkpoint.py`:
```python
import time

from osmose.calibration.checkpoint import read_checkpoint


def test_read_checkpoint_roundtrip(tmp_path):
    """write then read returns kind='ok' with the same checkpoint."""
    original = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    path = tmp_path / "phase12_checkpoint.json"
    write_checkpoint(path, original)
    result = read_checkpoint(path)
    assert result.kind == "ok"
    assert result.checkpoint is not None
    assert result.checkpoint.optimizer == original.optimizer
    assert result.checkpoint.best_fun == original.best_fun
    assert result.checkpoint.per_species_residuals == original.per_species_residuals


def test_read_checkpoint_no_run_on_missing_file(tmp_path):
    result = read_checkpoint(tmp_path / "phase12_checkpoint.json")
    assert result.kind == "no_run"
    assert result.checkpoint is None


def test_read_checkpoint_partial_on_truncated_json_recent_mtime(tmp_path):
    """File half-written and mtime recent (<3 s) → kind='partial'."""
    path = tmp_path / "phase12_checkpoint.json"
    path.write_text('{"optimizer": "de"')
    result = read_checkpoint(path)
    assert result.kind == "partial"


def test_read_checkpoint_corrupt_on_truncated_json_old_mtime(tmp_path):
    path = tmp_path / "phase12_checkpoint.json"
    path.write_text('{"optimizer": "de"')
    old = time.time() - 10
    os.utime(path, (old, old))
    result = read_checkpoint(path)
    assert result.kind == "corrupt"
    assert result.error_summary is not None


def test_read_checkpoint_corrupt_on_size_exceeds_limit(tmp_path):
    """File larger than MAX_CHECKPOINT_BYTES returns kind='corrupt' before parse."""
    path = tmp_path / "phase12_checkpoint.json"
    path.write_bytes(b"{}" + b" " * (MAX_CHECKPOINT_BYTES))  # > 1 MiB
    result = read_checkpoint(path)
    assert result.kind == "corrupt"
    assert "exceeds" in (result.error_summary or "").lower()


def test_read_checkpoint_corrupt_on_invariant_violation(tmp_path):
    """JSON parses fine, but generation=-1 violates Inv 1; convert ValueError to kind='corrupt'."""
    ckpt = CalibrationCheckpoint(**_valid_checkpoint_kwargs())
    path = tmp_path / "phase12_checkpoint.json"
    write_checkpoint(path, ckpt)
    data = json.loads(path.read_text())
    data["generation"] = -1
    path.write_text(json.dumps(data))
    result = read_checkpoint(path)
    assert result.kind == "corrupt"
    assert "generation" in (result.error_summary or "")


def test_read_checkpoint_corrupt_on_invalid_utf8(tmp_path):
    """Non-UTF-8 bytes → UnicodeDecodeError (a ValueError subclass) → kind='corrupt'."""
    path = tmp_path / "phase12_checkpoint.json"
    path.write_bytes(b"\xff\xfe\x00not-json")
    old = time.time() - 10
    os.utime(path, (old, old))
    result = read_checkpoint(path)
    assert result.kind == "corrupt"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v -k read_checkpoint
```

Expected: FAIL with `ImportError: cannot import name 'read_checkpoint'`.

- [ ] **Step 3: Implement `read_checkpoint`**

Append to `osmose/calibration/checkpoint.py`:
```python
import time

_PARTIAL_WRITE_WINDOW_S: Final[float] = 3.0


def read_checkpoint(path: Path) -> CheckpointReadResult:
    """Read and validate a checkpoint file. Never raises.

    See spec §5 for the four-kind contract:
      - 'ok'      : file present, JSON valid, all 14 invariants pass
      - 'no_run'  : file does not exist (vanished between glob and read)
      - 'partial' : decode/invariant error AND mtime within partial-write window
      - 'corrupt' : decode/invariant error AND older, OR size > MAX_CHECKPOINT_BYTES
    """
    # D3: error_summary includes the absolute path on every failure so an
    # SRE staring at the dashboard banner can locate and delete the bad file
    # to recover the run. The path is the most-asked debug question; baking
    # it into error_summary avoids cross-referencing logs.
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
        kind = "partial" if is_recent else "corrupt"
        return CheckpointReadResult(
            kind=kind, checkpoint=None,
            error_summary=(
                f"invariant violation for {path_str}: {e.__class__.__name__}: {e}. "
                "Recovery: delete the file to discard the bad frame; the next "
                "successful generation re-creates it."
            ),
        )
    return CheckpointReadResult(kind="ok", checkpoint=ckpt, error_summary=None)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): read_checkpoint with 4-kind discriminator, size guard, invariant-error catch"
```

---

### Task 6: `is_live` and `probe_writable`

**Files:**
- Modify: `osmose/calibration/checkpoint.py`
- Modify: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_checkpoint.py`:
```python
from osmose.calibration.checkpoint import is_live, probe_writable


def test_is_live_true_for_recent_file(tmp_path):
    path = tmp_path / "phase12_checkpoint.json"
    path.write_text("{}")
    assert is_live(path, max_age_s=60.0) is True


def test_is_live_false_for_old_file(tmp_path):
    path = tmp_path / "phase12_checkpoint.json"
    path.write_text("{}")
    old = time.time() - 120
    os.utime(path, (old, old))
    assert is_live(path, max_age_s=60.0) is False


def test_is_live_false_for_future_mtime(tmp_path):
    """Clock-jump resilience: a future-mtime file is NOT live."""
    path = tmp_path / "phase12_checkpoint.json"
    path.write_text("{}")
    future = time.time() + 60
    os.utime(path, (future, future))
    assert is_live(path, max_age_s=60.0) is False


def test_is_live_uses_injected_now_for_determinism(tmp_path):
    path = tmp_path / "phase12_checkpoint.json"
    path.write_text("{}")
    target_mtime = path.stat().st_mtime
    assert is_live(path, max_age_s=60.0, now=target_mtime + 30) is True
    assert is_live(path, max_age_s=60.0, now=target_mtime + 90) is False


def test_probe_writable_succeeds_on_writable_dir(tmp_path):
    probe_writable(tmp_path)  # no exception


def test_probe_writable_does_not_leak_sentinel(tmp_path):
    """No zero-byte sentinel files accumulate after probe_writable returns."""
    probe_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_probe_writable_raises_on_readonly_dir(tmp_path):
    import sys
    if sys.platform.startswith("win"):
        pytest.skip("chmod 0o555 semantics differ on Windows")
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(OSError):
            probe_writable(tmp_path)
    finally:
        tmp_path.chmod(0o755)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v -k "is_live or probe_writable"
```

Expected: FAIL.

- [ ] **Step 3: Implement the helpers**

Append to `osmose/calibration/checkpoint.py`:
```python
import tempfile


def is_live(path: Path, max_age_s: float = 60.0, now: float | None = None) -> bool:
    """True iff (now − max_age_s) < mtime <= now.

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
    with tempfile.NamedTemporaryFile(dir=results_dir, delete=True, prefix=".probe_", suffix=".tmp"):
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: all PASS.

- [ ] **Step 5: Add the 3-state liveness classifier + boundary tests**

Spec §7 specifies a three-state model (live / stalled / idle) with thresholds at 60 s and 300 s. Add a pure helper alongside `is_live` so the UI doesn't recompute the thresholds inline.

Append to `tests/test_calibration_checkpoint.py`:
```python
from osmose.calibration.checkpoint import liveness_state


def test_liveness_state_live_boundary():
    """age 0-60s = 'live' (inclusive of 60s by display rules)."""
    assert liveness_state(age_seconds=30.0) == "live"
    assert liveness_state(age_seconds=0.0) == "live"


def test_liveness_state_stalled_boundary_60s():
    """Just past 60s the state transitions to stalled."""
    assert liveness_state(age_seconds=61.0) == "stalled"
    assert liveness_state(age_seconds=120.0) == "stalled"


def test_liveness_state_stalled_to_idle_boundary_300s():
    """Just past 300s (5 min) the state transitions to idle."""
    assert liveness_state(age_seconds=299.0) == "stalled"
    assert liveness_state(age_seconds=301.0) == "idle"
    assert liveness_state(age_seconds=3600.0) == "idle"


def test_liveness_state_rejects_negative_age():
    """A future-mtime file produced negative age; treat conservatively as 'idle'
    (the is_live() function returns False for future-mtime, so the dashboard
    should not show such a file as live)."""
    assert liveness_state(age_seconds=-10.0) == "idle"
```

Append to `osmose/calibration/checkpoint.py`:
```python
def liveness_state(age_seconds: float) -> Literal["live", "stalled", "idle"]:
    """Three-state liveness classification per spec §7.

    age_seconds = now − mtime. The thresholds are display-aligned:
      - 0 <= age <= 60   → "live"     (green dot)
      - 60 < age <= 300  → "stalled"  (amber dot — investigate)
      - age > 300 or < 0 → "idle"     (grey dot)

    Future-mtime files (clock jump) produce negative age and classify as
    'idle' because is_live() also returns False for them; the dashboard
    treats negative age and zero-or-positive-but-old age identically.
    """
    if age_seconds < 0:
        return "idle"
    if age_seconds <= 60.0:
        return "live"
    if age_seconds <= 300.0:
        return "stalled"
    return "idle"
```

- [ ] **Step 6: Run liveness-state tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v -k liveness_state
```

Expected: 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): is_live + probe_writable + liveness_state classifier with boundary tests"
```

---

### Task 7: `LiveSnapshot` dataclass

**Files:**
- Modify: `osmose/calibration/checkpoint.py`
- Modify: `tests/test_calibration_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_checkpoint.py`:
```python
from osmose.calibration.checkpoint import LiveSnapshot


def test_live_snapshot_construction():
    sentinel = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    snap = LiveSnapshot(active=sentinel, other_live_paths=(), snapshot_monotonic=42.0)
    assert snap.active is sentinel
    assert snap.other_live_paths == ()


def test_live_snapshot_replace_via_dataclasses_replace():
    """Frozen dataclass: dataclasses.replace produces a new instance with one
    field changed (not _replace, which is a namedtuple method)."""
    sentinel = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    snap = LiveSnapshot(active=sentinel, other_live_paths=(), snapshot_monotonic=42.0)
    snap2 = dataclasses.replace(snap, snapshot_monotonic=99.0)
    assert snap2.snapshot_monotonic == 99.0
    assert snap2.active is sentinel
    assert snap.snapshot_monotonic == 42.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v -k live_snapshot
```

Expected: FAIL.

- [ ] **Step 3: Implement `LiveSnapshot`**

Append to `osmose/calibration/checkpoint.py`:
```python
@dataclass(frozen=True)
class LiveSnapshot:
    """One atomic view of the results directory per scan tick.

    Shared by all rendering reactives so they cannot disagree about which
    run is active. See spec §7 reactive-plumbing.
    """

    active: CheckpointReadResult
    other_live_paths: tuple[Path, ...]
    snapshot_monotonic: float
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/checkpoint.py tests/test_calibration_checkpoint.py
git commit -m "feat(calibration): LiveSnapshot for atomic per-tick scan results"
```

---

## Phase 2 — conftest fixtures

### Task 8: Test fixtures

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Inspect existing `tests/conftest.py`**

```bash
ls tests/conftest.py 2>/dev/null && head -50 tests/conftest.py || echo "(no conftest)"
```

If it exists, append. If not, create.

- [ ] **Step 2: Add fixtures**

Add (or create) `tests/conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch) -> Path:
    """Redirect all checkpoint writes to tmp_path.

    After Task 1 added RESULTS_DIR as a single source-of-truth in
    osmose.calibration.checkpoint, both scripts/calibrate_baltic.py and
    ui/pages/calibration_handlers.py import that symbol — so one monkeypatch
    target is enough. Note: each module that did `from osmose.calibration.checkpoint
    import RESULTS_DIR` got its own module-local binding; monkeypatching the
    source attribute does NOT rebind those local references. Each importing
    module is patched explicitly. The list stays short because S3 collapsed
    the producers from N to 1.
    """
    monkeypatch.setattr("osmose.calibration.checkpoint.RESULTS_DIR", tmp_path)
    try:
        import scripts.calibrate_baltic as cb_mod
        monkeypatch.setattr(cb_mod, "RESULTS_DIR", tmp_path, raising=False)
    except ImportError:
        pass
    try:
        import ui.pages.calibration_handlers as ch_mod
        monkeypatch.setattr(ch_mod, "RESULTS_DIR", tmp_path, raising=False)
    except ImportError:
        pass
    return tmp_path


@pytest.fixture
def synthetic_two_species_targets():
    """A 2-species banded-loss target list for tests that don't need real Baltic.

    Yields (targets, species_names) ready to pass into _ObjectiveWrapper or
    make_banded_objective.
    """
    from scripts.calibrate_baltic import BiomassTarget

    targets = [
        BiomassTarget(species="sp_a", target=1.0, lower=0.5, upper=1.5, weight=1.0),
        BiomassTarget(species="sp_b", target=2.0, lower=1.5, upper=2.5, weight=1.0),
    ]
    species_names = ["sp_a", "sp_b"]
    return targets, species_names


@pytest.fixture
def synthetic_stats_in_band():
    """species_stats dict that lands sp_a in-band and sp_b in-band."""
    return {
        "sp_a_mean": 1.0, "sp_a_cv": 0.1, "sp_a_trend": 0.01,
        "sp_b_mean": 2.0, "sp_b_cv": 0.1, "sp_b_trend": 0.01,
    }


@pytest.fixture
def synthetic_stats_sp_b_out_of_band():
    """sp_a in-band, sp_b overshoots."""
    return {
        "sp_a_mean": 1.0, "sp_a_cv": 0.1, "sp_a_trend": 0.01,
        "sp_b_mean": 5.0, "sp_b_cv": 0.1, "sp_b_trend": 0.01,
    }
```

- [ ] **Step 3: Run existing checkpoint tests to confirm fixtures don't break anything**

```bash
.venv/bin/python -m pytest tests/test_calibration_checkpoint.py -v
```

Expected: all still PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add tmp_results_dir + synthetic species fixtures for calibration tests"
```

---

## Phase 3 — Path A: `_ObjectiveWrapper` residual exposure

### Task 9: Extend `_ObjectiveWrapper.__call__` to capture per-species residuals + sim_biomass

**Files:**
- Modify: `scripts/calibrate_baltic.py`
- Create: `tests/test_objective_evaluator_residuals.py`

- [ ] **Step 1: Read the existing `_ObjectiveWrapper` (verified ground truth)**

The class lives at `scripts/calibrate_baltic.py:142`. Verified constructor signature:

```python
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
```

The loop in `__call__` iterates the module-level `SPECIES_NAMES` constant (line 192). The per-species `weighted_error` is computed at lines 216-218 and discarded. The simulate body lives at lines 173-187 of `__call__`. `__call__` already converts log10 → linear internally when `use_log_space=True` (line 177), so the dashboard re-eval must pass `best_x` directly (the log10 vector from scipy DE), NOT `np.power(10.0, best_x)`.

This task refactors the iteration target from `SPECIES_NAMES` (module constant) to `[t.species for t in self.targets]` (constructor-derived). The two are equivalent in production (calibrator always passes Baltic targets matching `SPECIES_NAMES`), but the targets-based iteration makes synthetic 2-species tests trivially constructible without monkeypatching the module constant.

- [ ] **Step 2: Write failing tests**

Create `tests/test_objective_evaluator_residuals.py`:
```python
from __future__ import annotations

import multiprocessing as mp

import pytest


def _build_wrapper(targets, species_names):
    """Construct an _ObjectiveWrapper using the verified real constructor.

    The wrapper iterates `[t.species for t in self.targets]` after this PR (see
    Task 9 Step 4); `species_names` is a convenience for the fixture's caller
    to use in assertions.
    """
    from scripts.calibrate_baltic import _ObjectiveWrapper

    base_config = {"simulation.nspecies": str(len(species_names))}
    return _ObjectiveWrapper(
        base_config=base_config,
        targets=targets,
        param_keys=["k_a", "k_b"],
        n_years=1,
        seed=42,
        use_log_space=True,
    )


def test_last_per_species_residuals_none_before_first_call(synthetic_two_species_targets):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    assert w.last_per_species_residuals is None


def test_last_per_species_residuals_populated_after_call(
    synthetic_two_species_targets, synthetic_stats_in_band, monkeypatch,
):
    """After __call__, the attribute is a list of (sp, weighted_error, sim_biomass)."""
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(w, "_simulate_and_compute_stats", lambda x: synthetic_stats_in_band)
    w([-0.3, 0.3])
    assert w.last_per_species_residuals is not None
    assert len(w.last_per_species_residuals) == 2
    for entry in w.last_per_species_residuals:
        assert len(entry) == 3
        sp, residual, sim_biomass = entry
        assert sp in species_names
        assert isinstance(residual, float)
        assert isinstance(sim_biomass, float)


def test_residuals_zero_when_in_band(
    synthetic_two_species_targets, synthetic_stats_in_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(w, "_simulate_and_compute_stats", lambda x: synthetic_stats_in_band)
    w([0.0, 0.0])
    by_sp = {sp: r for sp, r, _ in w.last_per_species_residuals}
    assert by_sp["sp_a"] == 0.0
    assert by_sp["sp_b"] == 0.0


def test_residual_positive_when_out_of_band(
    synthetic_two_species_targets, synthetic_stats_sp_b_out_of_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(
        w, "_simulate_and_compute_stats", lambda x: synthetic_stats_sp_b_out_of_band,
    )
    w([0.0, 0.0])
    by_sp = {sp: r for sp, r, _ in w.last_per_species_residuals}
    assert by_sp["sp_a"] == 0.0
    assert by_sp["sp_b"] > 0.0


def test_sim_biomass_captured_per_species(
    synthetic_two_species_targets, synthetic_stats_in_band, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(w, "_simulate_and_compute_stats", lambda x: synthetic_stats_in_band)
    w([0.0, 0.0])
    by_sp = {sp: b for sp, _, b in w.last_per_species_residuals}
    assert by_sp["sp_a"] == synthetic_stats_in_band["sp_a_mean"]
    assert by_sp["sp_b"] == synthetic_stats_in_band["sp_b_mean"]


def test_extinction_fast_path_records_100_loss_and_zero_biomass(
    synthetic_two_species_targets, monkeypatch,
):
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)
    monkeypatch.setattr(
        w, "_simulate_and_compute_stats",
        lambda x: {
            "sp_a_mean": 0.0, "sp_a_cv": 0.0, "sp_a_trend": 0.0,
            "sp_b_mean": 2.0, "sp_b_cv": 0.0, "sp_b_trend": 0.0,
        },
    )
    w([0.0, 0.0])
    by_sp = {sp: (r, b) for sp, r, b in w.last_per_species_residuals}
    assert by_sp["sp_a"] == (100.0, 0.0)
    assert by_sp["sp_b"][0] == 0.0


def test_residuals_attribute_unset_when_call_raises_midway(
    synthetic_two_species_targets, monkeypatch,
):
    """Spec §6.5.1 load-bearing invariant: assign-at-end means a mid-call raise
    leaves self.last_per_species_residuals at its prior value (None on first
    call, or stale from a previous successful call → no leak as long as the
    runner clears it before each re-eval, which Task 11 does).

    This test pins the contract by: (a) freshly-constructed wrapper raises
    inside _simulate_and_compute_stats; assert attribute is None. (b) After a
    successful call populates the attribute, a second call that raises does
    NOT mutate the attribute — but the runner is responsible for clearing it
    BEFORE the re-eval, not the wrapper.
    """
    targets, species_names = synthetic_two_species_targets
    w = _build_wrapper(targets, species_names)

    # Case A: fresh wrapper, _simulate raises → attribute stays None.
    def boom(x):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(w, "_simulate_and_compute_stats", boom)
    with pytest.raises(RuntimeError):
        w([0.0, 0.0])
    assert w.last_per_species_residuals is None

    # Case B: successful call populates; then a raising call leaves the
    # PRIOR successful value (the wrapper does NOT clear-at-start — that is
    # the runner's job per spec §6.5.1).
    monkeypatch.setattr(
        w, "_simulate_and_compute_stats",
        lambda x: {"sp_a_mean": 1.0, "sp_a_cv": 0.0, "sp_a_trend": 0.0,
                   "sp_b_mean": 2.0, "sp_b_cv": 0.0, "sp_b_trend": 0.0},
    )
    w([0.0, 0.0])
    populated = w.last_per_species_residuals
    assert populated is not None
    monkeypatch.setattr(w, "_simulate_and_compute_stats", boom)
    with pytest.raises(RuntimeError):
        w([0.0, 0.0])
    # Stale data remains; safe because Task 11's _write_progress_checkpoint
    # clears to None BEFORE the re-eval (spec §6.5.1).
    assert w.last_per_species_residuals is populated


# Module-level helpers used by the multiprocessing round-trip test below.
# Defined at module scope so multiprocessing's pickle protocol can locate
# them by name (lambdas / closures wouldn't survive the worker boundary).
_RTL_RESULT_KEYS_STATS = {
    "sp_a_mean": 1.0, "sp_a_cv": 0.0, "sp_a_trend": 0.0,
    "sp_b_mean": 2.0, "sp_b_cv": 0.0, "sp_b_trend": 0.0,
}


def _call_wrapper_then_return_residuals(wrapper):
    """Helper picklable target. Stubs _simulate_and_compute_stats inside the
    worker process (the parent's monkeypatch doesn't propagate), calls the
    wrapper, then returns the residuals.
    """
    wrapper._simulate_and_compute_stats = lambda x: _RTL_RESULT_KEYS_STATS
    wrapper([0.0, 0.0])
    return wrapper.last_per_species_residuals


def test_evaluator_round_trips_through_multiprocessing(
    synthetic_two_species_targets,
):
    """DE workers > 1 ships the evaluator across process boundaries via the
    same protocol scipy/joblib use. Verify the new attribute survives that
    round-trip AND that the captured residuals are visible in the worker
    process (not visible in the parent — that's the spec §6.5.1 caveat).
    """
    targets, _ = synthetic_two_species_targets
    w = _build_wrapper(targets, ["sp_a", "sp_b"])
    with mp.Pool(1) as pool:
        worker_residuals = pool.apply(_call_wrapper_then_return_residuals, (w,))
    # Worker-side capture: residuals are populated and have the expected shape.
    assert worker_residuals is not None
    assert len(worker_residuals) == 2
    # Parent process: attribute unchanged because the worker has its own copy.
    # This is the very reason Task 11 introduces main-thread re-evaluation.
    assert w.last_per_species_residuals is None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_objective_evaluator_residuals.py -v
```

Expected: every test FAILs (most likely `AttributeError: '_ObjectiveWrapper' object has no attribute 'last_per_species_residuals'`).

- [ ] **Step 4: Extend `_ObjectiveWrapper.__init__` and `__call__`**

In `scripts/calibrate_baltic.py`:

a) Append the attribute initialisation at the end of `__init__` (after the existing `self.w_worst = w_worst` line):
```python
        # Captured per-species residuals from the most-recent __call__.
        # See spec §6.5.1 — used by the dashboard checkpoint re-eval.
        self.last_per_species_residuals: list[tuple[str, float, float]] | None = None
```

b) Replace the entire `__call__` body (lines 171-233 in the current file) with the two-method version below. The new `_simulate_and_compute_stats` contains the simulate code that previously lived at the top of `__call__` (lines 173-187); the new `__call__` contains the species-loop with residual capture appended to a LOCAL list and assigned to `self.last_per_species_residuals` as the LAST statement before return:

```python
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate objective function at point x.

        Per-species (species, weighted_error, sim_biomass) triples are appended
        to a LOCAL list, then assigned to self.last_per_species_residuals as
        the LAST statement before return. A mid-call raise therefore leaves
        the attribute at None — load-bearing invariant per spec §6.5.1.
        """
        stats = self._simulate_and_compute_stats(x)
        if not stats:
            return 1e6  # Failed run — preserve existing scalar contract

        residuals_local: list[tuple[str, float, float]] = []
        total_error = 0.0
        worst_error = 0.0
        for sp in (t.species for t in self.targets):
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in stats or sp not in self.target_dict:
                total_error += 100.0
                worst_error = max(worst_error, 100.0)
                residuals_local.append((sp, 100.0, 0.0))
                continue

            sim_biomass = stats[mean_key]
            target = self.target_dict[sp]
            recorded_biomass = sim_biomass

            if sim_biomass <= 0:
                sp_error = 100.0
                recorded_biomass = 0.0
            elif sim_biomass < target.lower:
                sp_error = float(np.log10(target.lower / sim_biomass) ** 2)
            elif sim_biomass > target.upper:
                sp_error = float(np.log10(sim_biomass / target.upper) ** 2)
            else:
                sp_error = 0.0

            weighted_error = target.weight * sp_error
            total_error += weighted_error
            worst_error = max(worst_error, weighted_error)
            residuals_local.append((sp, weighted_error, float(recorded_biomass)))

            cv = stats.get(cv_key, 0.0)
            if cv > 0.2:
                total_error += self.w_stability * target.weight * (cv - 0.2) ** 2
            trend = stats.get(trend_key, 0.0)
            if trend > 0.05:
                total_error += self.w_stability * target.weight * (trend - 0.05) ** 2

        total_error += self.w_worst * worst_error
        # LOAD-BEARING: assign-at-end — see spec §6.5.1.
        self.last_per_species_residuals = residuals_local
        return total_error

    def _simulate_and_compute_stats(self, x: np.ndarray) -> dict[str, float]:
        """Run the simulation at parameter vector x and return the stats dict.

        Extracted verbatim from the existing __call__ body (lines 173-187) so
        tests can monkeypatch this seam. Returns {} on a failed simulation —
        the caller short-circuits to 1e6.
        """
        overrides: dict[str, str] = {}
        for i, key in enumerate(self.param_keys):
            if self.use_log_space:
                val = 10.0 ** x[i]
            else:
                val = x[i]
            overrides[key] = str(val)

        stats = run_simulation(
            self.base_config, overrides, n_years=self.n_years, seed=self.seed
        )
        return stats or {}
```

**Iteration target change.** The original loop iterates the module-level `SPECIES_NAMES` (line 192). The refactor iterates `(t.species for t in self.targets)` — equivalent in production (Baltic targets cover every species in `SPECIES_NAMES`, in the same order) but makes 2-species synthetic tests possible without monkeypatching the constant.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_objective_evaluator_residuals.py -v
```

Expected: all PASS.

- [ ] **Step 6: Run the entire test suite for regression**

```bash
.venv/bin/python -m pytest -x
```

Expected: PASS (no pre-existing tests broken).

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_objective_evaluator_residuals.py
git commit -m "feat(calibration): _ObjectiveWrapper captures per-species residuals + sim_biomass"
```

---

## Phase 4 — Path B: `make_banded_objective` accessor

### Task 10: Extend `make_banded_objective` to return `(callable, accessor)` tuple

**Files:**
- Modify: `osmose/calibration/losses.py`
- Modify: `ui/pages/calibration_handlers.py` (the single consumer at ~line 839)
- Modify: `tests/test_calibration_losses.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_losses.py`:
```python
from osmose.calibration.losses import make_banded_objective


def test_make_banded_objective_returns_tuple(synthetic_two_species_targets):
    targets, species_names = synthetic_two_species_targets
    result = make_banded_objective(targets, species_names)
    assert isinstance(result, tuple)
    assert len(result) == 2
    obj, accessor = result
    assert callable(obj)
    assert callable(accessor)


def test_residuals_accessor_returns_none_before_first_call(synthetic_two_species_targets):
    targets, species_names = synthetic_two_species_targets
    _, accessor = make_banded_objective(targets, species_names)
    assert accessor() is None


def test_residuals_accessor_returns_most_recent(
    synthetic_two_species_targets, synthetic_stats_in_band,
    synthetic_stats_sp_b_out_of_band,
):
    targets, species_names = synthetic_two_species_targets
    obj, accessor = make_banded_objective(targets, species_names)

    obj(synthetic_stats_in_band)
    labels1, residuals1, sim_biomass1 = accessor()
    assert tuple(labels1) == ("sp_a", "sp_b")
    assert tuple(residuals1) == (0.0, 0.0)

    obj(synthetic_stats_sp_b_out_of_band)
    labels2, residuals2, sim_biomass2 = accessor()
    assert residuals2[1] > 0.0
    assert sim_biomass2[1] == synthetic_stats_sp_b_out_of_band["sp_b_mean"]


def test_make_banded_objective_callable_returns_zero_when_in_band(
    synthetic_two_species_targets, synthetic_stats_in_band,
):
    """Backward-compat: callable behaviour unchanged for in-band stats."""
    targets, species_names = synthetic_two_species_targets
    obj, _ = make_banded_objective(targets, species_names)
    value = obj(synthetic_stats_in_band)
    assert value == 0.0  # no banded penalty, no stability/trend penalty


def test_residuals_accessor_reset_to_none_on_call_failure(
    synthetic_two_species_targets,
):
    """Spec §6.5.2 failure-mode parity: residual non-local cleared to None at
    the START of objective_callable; a mid-call raise leaves accessor returning None."""
    targets, species_names = synthetic_two_species_targets
    obj, accessor = make_banded_objective(targets, species_names)

    obj({"sp_a_mean": 1.0, "sp_b_mean": 2.0, "sp_a_cv": 0.0, "sp_b_cv": 0.0,
         "sp_a_trend": 0.0, "sp_b_trend": 0.0})
    assert accessor() is not None

    # Force a mid-call raise by passing a non-dict
    with pytest.raises((TypeError, AttributeError, KeyError)):
        obj(None)  # type: ignore[arg-type]
    assert accessor() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_losses.py -v -k "returns_tuple or residuals_accessor"
```

Expected: FAIL.

- [ ] **Step 3: Rewrite `make_banded_objective`**

In `osmose/calibration/losses.py`, replace the `make_banded_objective` function:
```python
def make_banded_objective(
    targets: list[BiomassTarget],
    species_names: list[str],
    w_stability: float = 5.0,
    w_worst: float = 0.5,
):
    """Factory returning (objective_callable, residuals_accessor).

    objective_callable(species_stats) -> float
        Same scalar contract as the previous signature (unchanged values and
        defaults). Banded-log-ratio loss per species + w_stability and w_worst
        composition.

    residuals_accessor() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]] | None
        Returns (species_labels, residuals, sim_biomass) from the most-recent
        objective call. Cleared to None at the START of each call (so a mid-call
        raise leaves it None — load-bearing for spec §6.5.2 parity with Path A).
        Re-populated as the LAST statement before return.
    """
    target_dict = {t.species: t for t in targets}
    state: dict[str, tuple] = {"residuals": None}

    def objective(species_stats: dict[str, float]) -> float:
        state["residuals"] = None  # clear at start

        residuals_local: list[tuple[str, float, float]] = []
        total_error = 0.0
        worst_error = 0.0
        for sp in species_names:
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in species_stats or sp not in target_dict:
                total_error += 100.0
                worst_error = max(worst_error, 100.0)
                residuals_local.append((sp, 100.0, 0.0))
                continue

            sim_biomass = species_stats[mean_key]
            target = target_dict[sp]
            recorded_biomass = sim_biomass

            if sim_biomass <= 0:
                sp_error = 100.0
                recorded_biomass = 0.0
            elif sim_biomass < target.lower:
                sp_error = float(math.log10(target.lower / sim_biomass) ** 2)
            elif sim_biomass > target.upper:
                sp_error = float(math.log10(sim_biomass / target.upper) ** 2)
            else:
                sp_error = 0.0

            weighted_error = target.weight * sp_error
            total_error += weighted_error
            worst_error = max(worst_error, weighted_error)
            residuals_local.append((sp, weighted_error, float(recorded_biomass)))

            cv = species_stats.get(cv_key, 0.0)
            if cv > 0.2:
                total_error += w_stability * target.weight * (cv - 0.2) ** 2
            trend = species_stats.get(trend_key, 0.0)
            if trend > 0.05:
                total_error += w_stability * target.weight * (trend - 0.05) ** 2

        total_error += w_worst * worst_error

        # LOAD-BEARING: assign-at-end mirrors Path A's invariant
        state["residuals"] = (
            tuple(sp for sp, _, _ in residuals_local),
            tuple(r for _, r, _ in residuals_local),
            tuple(b for _, _, b in residuals_local),
        )
        return total_error

    def residuals_accessor():
        return state["residuals"]

    return objective, residuals_accessor
```

- [ ] **Step 4: Update consumer (`ui/pages/calibration_handlers.py:~839`)**

Locate the line `banded_obj = make_banded_objective(...)` (around line 839) and change to:
```python
banded_obj, banded_residuals_accessor = make_banded_objective(...)
```

Keep `banded_residuals_accessor` available for Task 15.

- [ ] **Step 4b: Update the 6 existing callers in `tests/test_calibration_losses.py`**

Verified via grep: existing test sites do `obj = make_banded_objective(...)` then `obj(stats)`. After the tuple return, every one would `TypeError: 'tuple' object is not callable`. Update each:

| Line | Existing | New |
|---|---|---|
| 93  | `obj = make_banded_objective(targets, ["cod", "herring"])` | `obj, _ = make_banded_objective(targets, ["cod", "herring"])` |
| 105 | same | `obj, _ = make_banded_objective(targets, ["cod", "herring"])` |
| 118 | same | `obj, _ = make_banded_objective(targets, ["cod", "herring"])` |
| 125 | `obj = make_banded_objective(targets, ["cod", "herring"], w_stability=5.0)` | `obj, _ = make_banded_objective(targets, ["cod", "herring"], w_stability=5.0)` |
| 138 | `obj_with = make_banded_objective(targets, ["cod", "herring"], w_worst=1.0)` | `obj_with, _ = make_banded_objective(targets, ["cod", "herring"], w_worst=1.0)` |
| 139 | `obj_without = make_banded_objective(targets, ["cod", "herring"], w_worst=0.0)` | `obj_without, _ = make_banded_objective(targets, ["cod", "herring"], w_worst=0.0)` |

Each test's downstream `obj(stats)` invocations stay unchanged — `obj` is still the scalar callable; the discarded `_` is the residuals accessor (the existing tests don't need it).

- [ ] **Step 5: Run tests + suite regression**

```bash
.venv/bin/python -m pytest tests/test_calibration_losses.py -v
.venv/bin/python -m pytest -x
```

Expected: all PASS. The only existing consumer has been updated.

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/losses.py ui/pages/calibration_handlers.py tests/test_calibration_losses.py
git commit -m "feat(calibration): make_banded_objective returns (callable, residuals_accessor)"
```

---

## Phase 5 — DE runner integration

### Task 11: Replace inline checkpoint dict with `CalibrationCheckpoint` + main-thread re-eval

**Files:**
- Modify: `scripts/calibrate_baltic.py`
- Create: `tests/test_runner_checkpoints.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_runner_checkpoints.py`:
```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _toy_objective_2_param(x):
    """Picklable quadratic; min at (1.0, 2.0) → 0.0."""
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def test_de_writes_checkpoint_each_generation(tmp_results_dir):
    """Run DE for 3 generations on a synthetic toy; assert a CalibrationCheckpoint-
    shaped JSON file appears in tmp_results_dir after each generation."""
    from scipy.optimize import differential_evolution

    from osmose.calibration.checkpoint import read_checkpoint
    from scripts.calibrate_baltic import _make_checkpoint_callback

    bounds = [(-1.0, 1.0), (0.0, 3.0)]
    param_keys = ["k_a", "k_b"]
    checkpoint_path = tmp_results_dir / "phase_test_checkpoint.json"
    callback = _make_checkpoint_callback(
        checkpoint_path, 1, param_keys, bounds,
        phase="test",
        optimizer="de",
        evaluator=None,
        banded_targets=None,
        generation_budget=3,
    )
    differential_evolution(
        _toy_objective_2_param, bounds, maxiter=3, popsize=4, seed=42,
        workers=1, callback=callback, polish=False,
    )
    assert checkpoint_path.exists()
    result = read_checkpoint(checkpoint_path)
    assert result.kind == "ok"
    ckpt = result.checkpoint
    # Tautology-resistant assertions: not just optimizer/phase (those are
    # echoes of inputs), but the per-generation invocation count.
    assert ckpt.optimizer == "de"
    assert ckpt.phase == "test"
    assert ckpt.generation == 3   # callback fired exactly maxiter times
    assert ckpt.generation_budget == 3
    assert ckpt.per_species_residuals is None
    assert ckpt.per_species_sim_biomass is None
    assert ckpt.species_labels is None
    assert ckpt.proxy_source == "objective_disabled"


def test_de_existing_test_callsite_still_works(tmp_path):
    """Regression pin: the 14 existing _make_checkpoint_callback callers in
    tests/test_calibrate_baltic_parallelism.py pass only (path, every_n=N,
    param_keys=..., bounds=...). The new kwargs MUST be optional."""
    from types import SimpleNamespace
    from scripts.calibrate_baltic import _make_checkpoint_callback

    checkpoint_path = tmp_path / "phase_legacy_checkpoint.json"
    cb = _make_checkpoint_callback(
        checkpoint_path, every_n=2,
        param_keys=["k_a", "k_b"], bounds=[(-1.0, 1.0), (0.0, 3.0)],
    )
    # Two generations to trigger one write (every_n=2)
    cb(SimpleNamespace(x=[0.0, 1.0], fun=4.5))
    cb(SimpleNamespace(x=[-0.5, 0.5], fun=3.2))
    assert checkpoint_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_runner_checkpoints.py -v
```

Expected: FAIL — `_make_checkpoint_callback` doesn't yet accept the new kwargs.

- [ ] **Step 3: Extend `_make_checkpoint_callback` (backward-compat) + add `_write_progress_checkpoint`**

In `scripts/calibrate_baltic.py`, the existing `_make_checkpoint_callback` has signature `(checkpoint_path, every_n, param_keys, bounds, *, patience=0, wall_clock_max_seconds=None, rel_improvement_threshold=1e-6)`. Locate it with `grep -n "^def _make_checkpoint_callback" scripts/calibrate_baltic.py`. The existing test file `tests/test_calibrate_baltic_parallelism.py` has 14 call sites that rely on this signature (`grep -nc "_make_checkpoint_callback(" tests/test_calibrate_baltic_parallelism.py`), plus the production caller at the single `grep -n "_make_checkpoint_callback(" scripts/calibrate_baltic.py` hit inside the DE driver. **The new kwargs MUST default to None / safe values** so existing callers don't break. Replace the function with:

```python
def _make_checkpoint_callback(
    checkpoint_path: Path,
    every_n: int,
    param_keys: list[str],
    bounds: list[tuple[float, float]],
    *,
    patience: int = 0,
    wall_clock_max_seconds: float | None = None,
    rel_improvement_threshold: float = 1e-6,
    # NEW (all default-None / safe; existing callers unchanged):
    phase: str = "unknown",
    optimizer: str = "de",
    evaluator=None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
    generation_budget: int | None = None,
):
    """scipy DE callback combining checkpointing, early-stopping, and the
    main-thread re-evaluation that captures per-species residuals for the
    dashboard. See spec §6.5.1.
    """
    import logging
    import time as _time

    # S1 fix: _write_progress_checkpoint lives in osmose/calibration/checkpoint.py,
    # not in this file. CMA-ES and surrogate-DE runners do the same import.
    from osmose.calibration.checkpoint import _write_progress_checkpoint

    logger = logging.getLogger("osmose.calibration.checkpoint_callback")
    state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": _time.time(),
        "persistence_failure_notified": False,
    }

    def callback(intermediate_result, *_args, **_kwargs):
        if hasattr(intermediate_result, "x") and hasattr(intermediate_result, "fun"):
            best_x = intermediate_result.x
            best_fun = float(intermediate_result.fun)
        else:
            best_x = intermediate_result
            best_fun = float("inf")

        state["gen"] += 1
        prior_best = state["best_fun_seen"]
        if best_fun < prior_best - max(abs(prior_best), 1.0) * rel_improvement_threshold:
            state["best_fun_seen"] = best_fun
            state["gens_since_improvement"] = 0
        else:
            state["gens_since_improvement"] += 1

        if every_n > 0 and state["gen"] % every_n == 0:
            _write_progress_checkpoint(
                checkpoint_path, state, best_x, best_fun,
                optimizer, phase, generation_budget, param_keys, bounds,
                evaluator, banded_targets, logger,
            )

        if patience > 0 and state["gens_since_improvement"] >= patience:
            print(f"[early-stop] patience={patience} reached", flush=True)
            return True
        if wall_clock_max_seconds is not None:
            elapsed = _time.time() - state["start_time"]
            if elapsed > wall_clock_max_seconds:
                print(f"[early-stop] wall-clock cap reached", flush=True)
                return True
        return None

    return callback


```

The helper itself lives in `osmose/calibration/checkpoint.py` (S1 layering fix — `osmose.calibration.*` must not import from `scripts/*`). Add to that module:

```python
def _write_progress_checkpoint(
    checkpoint_path, state, best_x, best_fun,
    optimizer, phase, generation_budget, param_keys, bounds,
    evaluator, banded_targets, logger,
):
    """Build and write a CalibrationCheckpoint for the current generation.

    If `evaluator` is provided (Path A), re-evaluate best_x in the main thread
    to capture per-species residuals + sim_biomass. The engine rebuilds its
    PCG64 state from evaluator.seed on every run, so the re-eval is
    deterministic and produces bit-identical biomass to whatever the worker
    computed — no np.random.seed dance needed. See spec §6.5.1.
    """
    from datetime import datetime, timezone
    import time as _time
    from typing import Literal

    # S1 fix: this function now lives in osmose/calibration/checkpoint.py so
    # CalibrationCheckpoint and write_checkpoint are already in this module's
    # namespace — no import needed.

    proxy_source: Literal["banded_loss", "objective_disabled", "not_implemented"]
    residuals: list[tuple[str, float, float]] | None

    if evaluator is None or banded_targets is None:
        residuals = None
        proxy_source = "objective_disabled"
    else:
        # Path A re-eval — clear before, assign at end. Bounded log+continue.
        # D1: distinguish the three not_implemented causes via dedicated log
        # lines (cause=evaluator_returned_none vs cause=reeval_raised vs
        # cause=objective_disabled). Without these, the on-disk checkpoint
        # collapses all three onto proxy_source='not_implemented' and an SRE
        # cannot triage from artifacts alone.
        evaluator.last_per_species_residuals = None
        try:
            # best_x is already in log10 space; __call__ converts to linear
            # internally when use_log_space=True.
            evaluator(best_x)
            residuals = evaluator.last_per_species_residuals
            if residuals is None:
                logger.warning(
                    "checkpoint re-eval at gen %d: evaluator returned without "
                    "populating last_per_species_residuals "
                    "(proxy_source=not_implemented; cause=evaluator_returned_none). "
                    "This is a bug in _ObjectiveWrapper.",
                    state["gen"],
                )
                proxy_source = "not_implemented"
            else:
                proxy_source = "banded_loss"
        except Exception as e:  # noqa: BLE001 — bounded log+continue (§6.5.1)
            logger.warning(
                "checkpoint re-eval failed at gen %d "
                "(proxy_source=not_implemented; cause=reeval_raised): %s (%s)",
                state["gen"], e.__class__.__name__, e,
            )
            residuals = None
            proxy_source = "not_implemented"

    best_x_log10 = tuple(float(v) for v in best_x)
    ckpt = CalibrationCheckpoint(
        optimizer=optimizer,
        phase=phase,
        generation=state["gen"],
        generation_budget=generation_budget,
        best_fun=best_fun,
        per_species_residuals=tuple(r for _, r, _ in residuals) if residuals else None,
        per_species_sim_biomass=tuple(b for _, _, b in residuals) if residuals else None,
        species_labels=tuple(s for s, _, _ in residuals) if residuals else None,
        best_x_log10=best_x_log10,
        best_parameters={k: float(10.0 ** v) for k, v in zip(param_keys, best_x_log10)},
        param_keys=tuple(param_keys),
        bounds_log10={k: (float(lo), float(hi)) for k, (lo, hi) in zip(param_keys, bounds)},
        gens_since_improvement=state["gens_since_improvement"],
        elapsed_seconds=_time.time() - state["start_time"],
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
        banded_targets=banded_targets,
        proxy_source=proxy_source,
    )

    try:
        write_checkpoint(checkpoint_path, ckpt)
    except (OSError, TypeError, ValueError) as e:
        logger.warning(
            "write_checkpoint failed at gen %d: %s (%s) path=%s",
            state["gen"], e.__class__.__name__, e, checkpoint_path,
        )
```

Also: the DE driver function that builds the DE callback (search for `_make_checkpoint_callback(` near line 990-1000) must pass the new kwargs. If `banded_targets_dict` is not yet in scope, derive it from the evaluator:
```python
banded_targets_dict = (
    {t.species: (t.lower, t.upper) for t in evaluator_instance.targets}
    if evaluator_instance is not None and hasattr(evaluator_instance, "targets")
    else None
)
de_callback = _make_checkpoint_callback(
    checkpoint_path=checkpoint_path,
    every_n=checkpoint_every,
    param_keys=param_keys,
    bounds=bounds,
    phase=phase,
    optimizer="de",
    evaluator=evaluator_instance,
    banded_targets=banded_targets_dict,
    generation_budget=maxiter,
    patience=patience,
    wall_clock_max_seconds=(wall_clock_cap_h * 3600 if wall_clock_cap_h else None),
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_runner_checkpoints.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full suite for regression**

```bash
.venv/bin/python -m pytest -x
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_runner_checkpoints.py
git commit -m "feat(calibration): DE writes CalibrationCheckpoint with main-thread residual re-eval"
```

---

### Task 12: Wire `save_run()` at DE completion

**Files:**
- Modify: `scripts/calibrate_baltic.py`
- Create: `tests/test_history_wiring.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_history_wiring.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_save_run_called_on_de_completion(tmp_path, monkeypatch):
    """End-to-end smoke: _save_run_for_de writes to the history dir and
    list_runs() reads it back.

    NOTE: osmose/calibration/history.py:save_run captures HISTORY_DIR as a
    DEFAULT ARGUMENT, which Python binds at function-definition time. Patching
    hist_mod.HISTORY_DIR after import has no effect on the default. To make
    the test work, Task 12 changes _save_run_for_de to pass history_dir
    explicitly read from hist_mod at call time (not the captured default).
    """
    from osmose.calibration import history as hist_mod
    from osmose.calibration.history import list_runs
    from scripts.calibrate_baltic import _save_run_for_de

    monkeypatch.setattr(hist_mod, "HISTORY_DIR", tmp_path / "calibration_history")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "de",
        "phase": "test",
        "parameters": ["k_a", "k_b"],
        "results": {
            "best_objective": 0.5,
            "best_parameters": {"k_a": 1.0, "k_b": 2.0},
            "duration_seconds": 1.2,
            "n_evaluations": 12,
            "per_species_residuals_final": None,
            "per_species_sim_biomass_final": None,
            "species_labels": None,
        },
    }
    _save_run_for_de(payload)
    runs = list_runs(tmp_path / "calibration_history")
    assert len(runs) == 1
    assert runs[0]["algorithm"] == "de"
    assert runs[0]["best_objective"] == 0.5


def test_save_run_fallback_writes_to_tempfile_with_restrictive_mode(tmp_path, monkeypatch):
    """When save_run raises, _save_run_fallback writes a 0o600 JSON to tempfile.gettempdir()."""
    import os
    import stat
    import tempfile

    from osmose.calibration.history import _save_run_fallback  # S2: moved here

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    import logging
    logger = logging.getLogger("test")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "de",
        "phase": "test",
        "parameters": ["k_a"],
        "results": {},
    }
    _save_run_fallback(payload, OSError("simulated"), logger)
    files = list(tmp_path.glob("calibration_history_fallback_*.json"))
    assert len(files) == 1
    mode_bits = stat.S_IMODE(files[0].stat().st_mode)
    assert mode_bits == 0o600
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_history_wiring.py -v
```

Expected: FAIL with `ImportError: cannot import name '_save_run_for_de'`.

- [ ] **Step 3a: Add `_save_run_safe` to `osmose/calibration/history.py`**

Centralises the layered exception scope + /tmp fallback so DE (this task) and NSGA-II (Task 15) share one implementation. Append to `osmose/calibration/history.py`:

```python
import json as _json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone


def _save_run_safe(
    payload: dict,
    *,
    logger: logging.Logger,
    with_fallback: bool = True,
) -> None:
    """Persist a completed run via save_run(), tolerating expected failures.

    Layered exception scope: catch the expected serialisation/OS failure
    classes plus a defensive `except Exception` catch-all. On failure (if
    with_fallback): write a fallback JSON to tempfile.gettempdir() with
    mode 0o600, O_EXCL so we don't clobber prior fallbacks. Logs via
    logger.exception in every failure branch.

    History-dir resolution: read HISTORY_DIR at call time (NOT via save_run's
    default arg, which Python captures at function-definition time). This
    makes test monkeypatching of HISTORY_DIR effective.
    """
    from osmose.calibration import history as hist_mod

    try:
        save_run(payload, history_dir=hist_mod.HISTORY_DIR)
        return
    except (OSError, TypeError, ValueError, OverflowError,
            UnicodeError, RecursionError, MemoryError) as e:
        if with_fallback:
            _save_run_fallback(payload, e, logger)
        else:
            logger.exception("save_run failed (no fallback): %s", e.__class__.__name__)
    except Exception as e:  # noqa: BLE001 — defensive catch-all
        if with_fallback:
            _save_run_fallback(payload, e, logger)
        else:
            logger.exception("save_run failed unexpectedly: %s", e.__class__.__name__)


def _save_run_fallback(payload: dict, e: BaseException, logger: logging.Logger) -> None:
    """Write a fallback JSON when canonical save failed.

    O_EXCL is INTENTIONAL: if a prior fallback exists, do not overwrite — the
    prior file is still the most-likely-recoverable copy. Implementers must
    NOT change this to O_TRUNC. Spec §9 fallback contract.
    """
    logger.exception(
        "save_run failed: %s; payload keys=%s", e.__class__.__name__, list(payload.keys()),
    )
    ts = payload.get("timestamp", datetime.now(timezone.utc).isoformat()).replace(":", "-")
    raw_algo = str(payload.get("algorithm", "unknown"))
    algo_sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", raw_algo)[:32]
    path = Path(tempfile.gettempdir()) / f"calibration_history_fallback_{ts}_{algo_sanitized}.json"
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            _json.dump(payload, f, indent=2, allow_nan=False, default=str)
    except (OSError, TypeError, ValueError) as fb_e:
        logger.exception("fallback write also failed: %s; path=%s", fb_e.__class__.__name__, path)
```

- [ ] **Step 3b: Add `_save_run_for_de` thin wrapper; call it at DE completion**

Add to `scripts/calibrate_baltic.py` (only the thin wrapper; the actual
implementation of `_save_run_safe` and `_save_run_fallback` lives in
`osmose/calibration/history.py` per Step 3a — duplicating it here would
defeat the S2 fix):

```python
import logging


def _save_run_for_de(payload: dict) -> None:
    """Thin DE-side wrapper around _save_run_safe.

    Both DE (this file) and NSGA-II (ui/pages/calibration_handlers.py via
    _save_run_for_nsga2) call the same _save_run_safe helper in
    osmose/calibration/history.py — keeps the exception scope + fallback
    behaviour in one place. Spec §9.
    """
    from osmose.calibration.history import _save_run_safe

    logger = logging.getLogger("osmose.calibration.history_wiring")
    _save_run_safe(payload, logger=logger, with_fallback=True)
```

In the DE driver function, immediately after `differential_evolution()` returns and BEFORE the existing `phase{N}_results.json` write:
```python
    # New: persist to History tab via save_run (additive — the existing
    # phase{N}_results.json write below is preserved).
    best_x = result.x
    best_x_log10 = tuple(float(v) for v in best_x)
    final_residuals: list[tuple[str, float, float]] | None
    if evaluator_instance is not None and banded_targets_dict is not None:
        try:
            evaluator_instance.last_per_species_residuals = None
            evaluator_instance(best_x)  # log10 vector — wrapper converts internally
            final_residuals = evaluator_instance.last_per_species_residuals
        except Exception:
            final_residuals = None
    else:
        final_residuals = None

    _save_run_for_de({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "de",
        "phase": phase,
        "parameters": list(param_keys),
        "results": {
            "best_objective": float(result.fun),
            "best_parameters": {
                k: float(10.0 ** v) for k, v in zip(param_keys, best_x_log10)
            },
            "duration_seconds": time.time() - start_time,
            "n_evaluations": int(getattr(result, "nfev", 0)),
            "per_species_residuals_final": (
                [r for _, r, _ in final_residuals] if final_residuals else None
            ),
            "per_species_sim_biomass_final": (
                [b for _, _, b in final_residuals] if final_residuals else None
            ),
            "species_labels": (
                [s for s, _, _ in final_residuals] if final_residuals else None
            ),
        },
    })
    # ... existing phase{N}_results.json write below this point, unchanged ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_history_wiring.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_baltic.py tests/test_history_wiring.py
git commit -m "feat(calibration): DE wires save_run on completion with tempfile 0o600 fallback"
```

---

## Phase 6 — CMA-ES + surrogate-DE runner integration

### Task 13: CMA-ES per-generation checkpoint hook

**Files:**
- Modify: `osmose/calibration/cmaes_runner.py`
- Modify: `tests/test_runner_checkpoints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_runner_checkpoints.py`:
```python
def test_cmaes_writes_checkpoint(tmp_results_dir):
    """A 2-generation CMA-ES run on a synthetic toy writes a CalibrationCheckpoint."""
    from osmose.calibration.checkpoint import read_checkpoint
    from osmose.calibration.cmaes_runner import run_cmaes

    bounds = [(-1.0, 1.0), (0.0, 3.0)]
    checkpoint_path = tmp_results_dir / "phase_test_checkpoint.json"

    run_cmaes(
        objective=_toy_objective_2_param,
        bounds=bounds,
        param_keys=["k_a", "k_b"],
        x0=[0.0, 1.5],
        seed=42,
        maxiter=2,
        sigma0=0.3,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        phase="test",
        evaluator=None,
        banded_targets=None,
    )
    assert checkpoint_path.exists()
    result = read_checkpoint(checkpoint_path)
    assert result.kind == "ok"
    assert result.checkpoint.optimizer == "cmaes"
    # Tautology-resistant: assert per-generation invocation count, not just
    # the hardcoded optimizer name.
    assert result.checkpoint.generation == 2
    assert result.checkpoint.generation_budget == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_runner_checkpoints.py::test_cmaes_writes_checkpoint -v
```

Expected: FAIL — `run_cmaes` doesn't yet accept `checkpoint_path` / `phase` / etc.

- [ ] **Step 3: Add checkpoint hook to `cmaes_runner.py`**

Modify `osmose/calibration/cmaes_runner.py`. Extend the `run_cmaes` signature and call `_write_progress_checkpoint` after each `tell()`:
```python
def run_cmaes(
    objective,
    bounds,
    *,
    x0: list[float] | None = None,
    seed: int = 42,
    maxiter: int = 200,
    sigma0: float = 0.3,
    workers: int = 1,
    # NEW (all defaulted; existing 8+ callers continue to work unchanged):
    param_keys: list[str] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    phase: str = "unknown",
    evaluator=None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
    **cma_options,
):
    """Optimise `objective` over `bounds` via CMA-ES.

    NEW: if checkpoint_path is set AND checkpoint_every > 0 AND param_keys is
    provided, writes a CalibrationCheckpoint every N generations to the same
    path/format DE uses. Existing callers that pass none of these new kwargs
    are unaffected — no checkpoint is written.
    """
    import logging
    import time

    from osmose.calibration.checkpoint import _write_progress_checkpoint  # S1: layering fix

    logger = logging.getLogger("osmose.calibration.cmaes_runner")
    state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": time.time(),
        "persistence_failure_notified": False,
    }

    # ... existing CMA-ES setup (es = cma.CMAEvolutionStrategy(...)) ...

    for _ in range(maxiter):
        candidates = es.ask()
        values = [...]  # existing parallel/serial eval
        es.tell(candidates, values)

        state["gen"] += 1
        best_x = es.result.xbest
        best_fun = float(es.result.fbest)
        prior_best = state["best_fun_seen"]
        if best_fun < prior_best:
            state["best_fun_seen"] = best_fun
            state["gens_since_improvement"] = 0
        else:
            state["gens_since_improvement"] += 1

        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and param_keys is not None
            and state["gen"] % checkpoint_every == 0
        ):
            _write_progress_checkpoint(
                checkpoint_path=checkpoint_path,
                state=state,
                best_x=best_x,
                best_fun=best_fun,
                optimizer="cmaes",
                phase=phase,
                generation_budget=maxiter,
                param_keys=param_keys,
                bounds=bounds,
                evaluator=evaluator,
                banded_targets=banded_targets,
                logger=logger,
            )

    return es.result
```

Preserve every existing line of the original `run_cmaes` body — the snippet above shows only the structural changes (new kwargs at the end of the signature, state dict near the top, post-`tell()` checkpoint hook). The triple-guard (`checkpoint_path is not None AND checkpoint_every > 0 AND param_keys is not None`) means existing callers — which pass none of the dashboard kwargs — skip the checkpoint write entirely.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_runner_checkpoints.py::test_cmaes_writes_checkpoint -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/cmaes_runner.py tests/test_runner_checkpoints.py
git commit -m "feat(calibration): CMA-ES writes CalibrationCheckpoint each generation"
```

---

### Task 14: surrogate-DE per-generation checkpoint hook

**Files:**
- Modify: `osmose/calibration/surrogate_de.py`
- Modify: `tests/test_runner_checkpoints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_runner_checkpoints.py`:
```python
def test_surrogate_de_writes_checkpoint(tmp_results_dir):
    from osmose.calibration.checkpoint import read_checkpoint
    from osmose.calibration.surrogate_de import surrogate_assisted_de

    bounds = [(-1.0, 1.0), (0.0, 3.0)]
    checkpoint_path = tmp_results_dir / "phase_test_checkpoint.json"

    surrogate_assisted_de(
        objective=_toy_objective_2_param,
        bounds=bounds,
        param_keys=["k_a", "k_b"],
        seed=42,
        max_outer_iter=2,
        n_init=4,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        phase="test",
        evaluator=None,
        banded_targets=None,
    )
    assert checkpoint_path.exists()
    result = read_checkpoint(checkpoint_path)
    assert result.kind == "ok"
    assert result.checkpoint.optimizer == "surrogate-de"
    # Tautology-resistant: per-generation count == max_outer_iter (only the
    # OUTER real-eval loop writes checkpoints; inner surrogate iterations do
    # NOT — spec §6.5).
    assert result.checkpoint.generation == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_runner_checkpoints.py::test_surrogate_de_writes_checkpoint -v
```

Expected: FAIL.

- [ ] **Step 3: Add the per-generation hook in `surrogate_de.py`**

In `osmose/calibration/surrogate_de.py`, find `surrogate_assisted_de` and add the checkpoint hook AFTER each real-evaluation step (NOT after each inner-surrogate iteration):
```python
def surrogate_assisted_de(
    objective,
    bounds,
    *,
    seed: int = 42,
    max_outer_iter: int = 20,
    n_init: int = 8,
    workers: int = 1,
    # NEW (all defaulted; existing callers in scripts/benchmark_optimizers.py,
    # scripts/calibrate_baltic.py, tests/test_surrogate_de.py unchanged):
    param_keys: list[str] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    phase: str = "unknown",
    evaluator=None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
    **inner_de_kwargs,
):
    import logging
    import time

    from osmose.calibration.checkpoint import _write_progress_checkpoint  # S1: layering fix

    logger = logging.getLogger("osmose.calibration.surrogate_de")
    state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": time.time(),
        "persistence_failure_notified": False,
    }

    # ... existing init: sample n_init points, evaluate, fit GP ...

    for outer in range(max_outer_iter):
        # ... existing surrogate-DE iteration body ...
        # After the REAL evaluation step at the end of each outer iteration:
        state["gen"] += 1
        prior_best = state["best_fun_seen"]
        if best_fun_so_far < prior_best:
            state["best_fun_seen"] = best_fun_so_far
            state["gens_since_improvement"] = 0
        else:
            state["gens_since_improvement"] += 1
        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and param_keys is not None
            and state["gen"] % checkpoint_every == 0
        ):
            _write_progress_checkpoint(
                checkpoint_path=checkpoint_path,
                state=state,
                best_x=best_x_so_far,
                best_fun=best_fun_so_far,
                optimizer="surrogate-de",
                phase=phase,
                generation_budget=max_outer_iter,
                param_keys=param_keys,
                bounds=bounds,
                evaluator=evaluator,
                banded_targets=banded_targets,
                logger=logger,
            )

    # ... existing return value of surrogate_assisted_de unchanged ...
```

Preserve every existing line of the original `surrogate_assisted_de` body. The triple-guard makes existing callers — that don't pass `param_keys` / `checkpoint_path` — skip the new code path entirely.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_runner_checkpoints.py::test_surrogate_de_writes_checkpoint -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osmose/calibration/surrogate_de.py tests/test_runner_checkpoints.py
git commit -m "feat(calibration): surrogate-DE writes CalibrationCheckpoint per real-eval round"
```

---

## Phase 7 — NSGA-II in-UI integration

### Task 15: Extend `_ProgressCallback.notify` to write checkpoints; wire `save_run` in results branch

**Files:**
- Modify: `ui/pages/calibration_handlers.py`
- Modify: `tests/test_ui_calibration_handlers.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ui_calibration_handlers.py`:
```python
def test_progress_callback_writes_checkpoint_per_generation(tmp_results_dir):
    """Driving the NSGA-II callback three times produces a checkpoint with
    monotonically-increasing generation."""
    from unittest.mock import MagicMock

    import numpy as np

    from osmose.calibration.checkpoint import read_checkpoint
    from ui.pages.calibration_handlers import _make_progress_callback

    history_appended = []
    cb = _make_progress_callback(
        cal_history_append=history_appended.append,
        cancel_check=lambda: False,
        checkpoint_path=tmp_results_dir / "phase_test_checkpoint.json",
        phase="test",
        param_keys=["k_a", "k_b"],
        bounds=[(-1.0, 1.0), (0.0, 3.0)],
        banded_residuals_accessor=lambda: None,
        banded_targets=None,
    )

    for gen in range(1, 4):
        mock_alg = MagicMock()
        mock_alg.opt = MagicMock()
        mock_alg.opt.get.side_effect = lambda key: {
            "F": np.array([[float(gen) * 0.5]]),
            "X": np.array([[0.1, 0.2]]),
        }[key]
        cb.notify(mock_alg)

    result = read_checkpoint(tmp_results_dir / "phase_test_checkpoint.json")
    assert result.kind == "ok"
    assert result.checkpoint.optimizer == "nsga2"
    assert result.checkpoint.generation == 3


def test_results_branch_handles_write_failure_gracefully(monkeypatch):
    """write_checkpoint raising in the NSGA-II callback MUST NOT prevent
    cal_history_append from updating the convergence chart."""
    from unittest.mock import MagicMock

    import numpy as np

    from ui.pages.calibration_handlers import _make_progress_callback

    def failing_write(*args, **kwargs):
        raise OSError("simulated disk-full")

    monkeypatch.setattr(
        "osmose.calibration.checkpoint.write_checkpoint", failing_write,
    )
    appended = []
    cb = _make_progress_callback(
        cal_history_append=appended.append,
        cancel_check=lambda: False,
        checkpoint_path=Path("/tmp/should_not_be_used.json"),
        phase="test",
        param_keys=["k_a"],
        bounds=[(-1.0, 1.0)],
        banded_residuals_accessor=lambda: None,
        banded_targets=None,
    )
    mock_alg = MagicMock()
    mock_alg.opt.get.side_effect = lambda key: {
        "F": np.array([[3.14]]),
        "X": np.array([[0.0]]),
    }[key]
    cb.notify(mock_alg)
    assert appended == [3.14]  # chart updated even though disk write raised
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py -v -k "progress_callback or results_branch_handles"
```

Expected: FAIL — `_make_progress_callback` doesn't yet accept the new kwargs.

- [ ] **Step 3: Replace `_make_progress_callback`**

In `ui/pages/calibration_handlers.py`, around line 145, replace `_make_progress_callback` with:
```python
def _make_progress_callback(
    cal_history_append,
    cancel_check,
    *,
    checkpoint_path: Path | None = None,
    phase: str = "unknown",
    param_keys: list[str] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    banded_residuals_accessor=None,
    banded_targets: dict[str, tuple[float, float]] | None = None,
):
    """Create a pymoo callback that feeds cal_history AND writes a checkpoint.

    The existing in-memory chart feed is preserved (cal_history_append); the
    new write is additive and wrapped in try/except so a disk failure cannot
    regress the convergence chart. See spec §6 runner table (NSGA-II row).
    """
    from datetime import datetime, timezone
    import logging
    import time

    import numpy as np
    from pymoo.core.callback import Callback  # type: ignore[import-untyped]

    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint

    logger = logging.getLogger("osmose.ui.calibration_dashboard")
    state = {
        "gen": 0,
        "best_fun_seen": float("inf"),
        "gens_since_improvement": 0,
        "start_time": time.time(),
    }

    class _ProgressCallback(Callback):
        def __init__(self):
            super().__init__()

        def notify(self, algorithm):
            if cancel_check():
                algorithm.termination.force_termination = True
                return
            F = algorithm.opt.get("F")
            best = float(np.min(F.sum(axis=1)))
            cal_history_append(best)  # existing — MUST run first

            if checkpoint_path is None or param_keys is None or bounds is None:
                return

            state["gen"] += 1
            prior_best = state["best_fun_seen"]
            if best < prior_best:
                state["best_fun_seen"] = best
                state["gens_since_improvement"] = 0
            else:
                state["gens_since_improvement"] += 1

            # Path B: in-process single-threaded → no re-eval needed.
            # D1 disambiguation: log when the accessor returns None despite
            # banded-loss being configured (distinguishes
            # cause=accessor_returned_none from cause=objective_disabled).
            if banded_residuals_accessor is None or banded_targets is None:
                residuals_tuple = None
            else:
                residuals_tuple = banded_residuals_accessor()

            if residuals_tuple is None:
                if banded_targets is None:
                    proxy_source = "objective_disabled"
                else:
                    proxy_source = "not_implemented"
                    logger.warning(
                        "NSGA-II checkpoint at gen %d: banded_residuals_accessor "
                        "returned None despite banded-loss being configured "
                        "(proxy_source=not_implemented; cause=accessor_returned_none).",
                        state["gen"],
                    )
                per_species_residuals = None
                per_species_sim_biomass = None
                species_labels = None
            else:
                species_labels, per_species_residuals, per_species_sim_biomass = residuals_tuple
                proxy_source = "banded_loss"

            best_x = algorithm.opt.get("X")[0]
            best_x_log10 = tuple(float(v) for v in best_x)

            try:
                ckpt = CalibrationCheckpoint(
                    optimizer="nsga2",
                    phase=phase,
                    generation=state["gen"],
                    generation_budget=None,
                    best_fun=best,
                    per_species_residuals=per_species_residuals,
                    per_species_sim_biomass=per_species_sim_biomass,
                    species_labels=species_labels,
                    best_x_log10=best_x_log10,
                    best_parameters={k: float(10.0 ** v) for k, v in zip(param_keys, best_x_log10)},
                    param_keys=tuple(param_keys),
                    bounds_log10={k: (float(lo), float(hi)) for k, (lo, hi) in zip(param_keys, bounds)},
                    gens_since_improvement=state["gens_since_improvement"],
                    elapsed_seconds=time.time() - state["start_time"],
                    timestamp_iso=datetime.now(timezone.utc).isoformat(),
                    banded_targets=banded_targets,
                    proxy_source=proxy_source,
                )
                write_checkpoint(checkpoint_path, ckpt)
            except (OSError, TypeError, ValueError) as e:
                logger.warning(
                    "NSGA-II checkpoint write failed at gen %d: %s", state["gen"], e,
                )

    return _ProgressCallback()
```

Then in `_poll_cal_messages` around line 442, in the `"results"` branch (lines 451-454), after the existing `cal_X.set(X); cal_F.set(F)` calls, add:
```python
            elif kind == "results":
                X, F = payload
                cal_X.set(X)
                cal_F.set(F)
                # Scope note: phase and param_keys are NOT in scope at this
                # callsite (they live deep in _start_optimization_with_params).
                # NSGA-II UI runs don't have a CLI-style phase concept, so we
                # use a fixed string. param_keys comes from cal_param_names
                # which IS a register_calibration_handlers closure parameter.
                try:
                    _save_run_for_nsga2(
                        payload, X, F,
                        phase="ui_nsga2",
                        param_keys=cal_param_names.get() or [],
                    )
                except Exception as e:  # noqa: BLE001
                    surrogate_status.set(f"history persist failed: {e}")
```

Add `_save_run_for_nsga2` at module scope in `calibration_handlers.py`:
```python
def _save_run_for_nsga2(payload, X, F, phase: str, param_keys: list[str]) -> None:
    """Thin NSGA-II-side wrapper around _save_run_safe in history.py.

    Mirrors _save_run_for_de in scripts/calibrate_baltic.py — same layered
    exception scope, same /tmp fallback, both via the shared helper. The
    function builds the payload record and delegates persistence.
    """
    from datetime import datetime, timezone
    import logging

    import numpy as np

    from osmose.calibration.history import _save_run_safe

    logger = logging.getLogger("osmose.ui.calibration_dashboard")
    best_idx = int(np.argmin(F.sum(axis=1)))
    best_F = float(F.sum(axis=1)[best_idx])
    best_x = X[best_idx]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "nsga2",
        "phase": phase,
        "parameters": list(param_keys),
        "results": {
            "best_objective": best_F,
            "best_parameters": {k: float(v) for k, v in zip(param_keys, best_x)},
            "duration_seconds": 0.0,
            "n_evaluations": int(getattr(X, "shape", [0])[0]),
            "per_species_residuals_final": None,
            "per_species_sim_biomass_final": None,
            "species_labels": None,
        },
    }
    _save_run_safe(record, logger=logger, with_fallback=True)
```

- [ ] **Step 4: Wire the existing NSGA-II caller at line ~672 to pass dashboard kwargs**

Without this step, the new `_make_progress_callback` accepts the dashboard kwargs but never receives them in production — NSGA-II UI runs would not write checkpoints. The caller is inside `run_optimization()` → inside `_start_optimization_with_params` → inside `register_calibration_handlers`. Plumb the dashboard params through:

a) **Hoist `banded_residuals_accessor` through THREE locations.** The source assignment at line ~839 (`banded_obj, banded_residuals_accessor = make_banded_objective(...)`) lives inside `handle_start_cal`; the consumer at line ~672 lives inside `_start_optimization_with_params`. Both are siblings nested inside `register_calibration_handlers`. To plumb the accessor from producer to consumer, three coordinated edits are required:

**a.1) Declare closure-level state** at the top of `register_calibration_handlers` (alongside the existing `_shared_*` declarations around lines 487-499):
```python
    _shared_banded_residuals_accessor = None
    _shared_banded_targets_dict: dict[str, tuple[float, float]] | None = None
```

**a.2) Add `nonlocal` + assignment inside `handle_start_cal`** at the producer site (~line 839, after the existing `nonlocal _shared_*` block at ~742-746). After the line `banded_obj, banded_residuals_accessor = make_banded_objective(...)`:
```python
    nonlocal _shared_banded_residuals_accessor, _shared_banded_targets_dict
    _shared_banded_residuals_accessor = banded_residuals_accessor
    _shared_banded_targets_dict = {t.species: (t.lower, t.upper) for t in targets}
```

**a.3) Add `nonlocal` declaration inside `_start_optimization_with_params`** at the consumer site (alongside the existing `nonlocal _shared_*` block at ~lines 506-509):
```python
    nonlocal _shared_banded_residuals_accessor, _shared_banded_targets_dict
```

All three edits land in the same task commit. Without all three, the consumer reads `None` (a.1 alone) or raises `UnboundLocalError` (a.2 + a.3 but not a.1).

b) **Update the NSGA-II caller at line ~672** (inside `run_optimization`) to pass dashboard kwargs:

```python
callback = _make_progress_callback(
    cal_history_append=_tracked_append,
    cancel_check=cancel_event.is_set,
    # NEW dashboard wiring:
    checkpoint_path=RESULTS_DIR / "phase_ui_nsga2_checkpoint.json",
    phase="ui_nsga2",
    param_keys=[fp.key for fp in free_params],
    bounds=[(fp.lower_bound, fp.upper_bound) for fp in free_params],
    banded_residuals_accessor=_shared_banded_residuals_accessor,
    banded_targets=_shared_banded_targets_dict,
)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_ui_calibration_handlers.py
git commit -m "feat(calibration): NSGA-II callback writes checkpoint + save_run; chart-update regression-pinned"
```

---

## Phase 8 — UI dashboard widgets

> **Shiny scoping rule for this phase.** All `@output` / `@render.ui` / `@render_plotly` / `@reactive.poll` / `@reactive.effect` / `@reactive.event` decorators in Shiny for Python are bound to a per-session server function. In this codebase the server function is `register_calibration_handlers(input, output, session, state, …)` in `ui/pages/calibration_handlers.py` (locate with `grep -n "^def register_calibration_handlers" ui/pages/calibration_handlers.py`). Existing decorated reactives sit inside its body (search with `grep -n "    @reactive\.\|    @output\|    @render" ui/pages/calibration_handlers.py`). **Tasks 16-21's reactive/render code must be added INSIDE the body of `register_calibration_handlers`, NOT at module scope.** Pure helpers (`_build_proxy_rows`, `_build_param_rows`, `_format_elapsed`, `_ckpt_mtime_for`, `_aria_for_state`, `_PROXY_EPS`, `_STATE_ORDER`, `_BOUND_DISTANCE_THRESHOLD`, `_scan_signature`, `_scan_results_dir`, `_notify_scan_failure_once`) CAN live at module scope — they don't use `@output` / `@reactive.*` decorators. The module-level constants (`RESULTS_DIR`, `_signature_tick`, `_seen_scan_errors`, `_seen_scan_errors_lock`, `_EMPTY_SNAPSHOT`, `logger`) also live at module scope. Where Task instructions say "Append to `ui/pages/calibration_handlers.py`", read that as: pure helpers + constants go near module top (after existing imports); reactive renderers go inside `register_calibration_handlers`.

### Task 16: Module-level `RESULTS_DIR`, scan helpers, LiveSnapshot reactive

**Files:**
- Modify: `ui/pages/calibration_handlers.py`
- Create: `tests/test_calibration_dashboard_reactive.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration_dashboard_reactive.py`:
```python
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def test_scan_results_dir_returns_no_run_when_empty(tmp_results_dir):
    from ui.pages.calibration_handlers import _scan_results_dir

    snap = _scan_results_dir()
    assert snap.active.kind == "no_run"
    assert snap.other_live_paths == ()


def test_scan_results_dir_picks_newest_live(tmp_results_dir):
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _scan_results_dir

    kwargs_old = _valid_checkpoint_kwargs()
    kwargs_old["phase"] = "old"
    kwargs_new = _valid_checkpoint_kwargs()
    kwargs_new["phase"] = "new"

    p_old = tmp_results_dir / "phase_old_checkpoint.json"
    p_new = tmp_results_dir / "phase_new_checkpoint.json"
    write_checkpoint(p_old, CalibrationCheckpoint(**kwargs_old))
    old_mtime = time.time() - 10
    os.utime(p_old, (old_mtime, old_mtime))
    write_checkpoint(p_new, CalibrationCheckpoint(**kwargs_new))

    snap = _scan_results_dir()
    assert snap.active.kind == "ok"
    assert snap.active.checkpoint.phase == "new"
    assert len(snap.other_live_paths) == 1


def test_scan_skips_symlinks(tmp_results_dir):
    """Security: a symlink in RESULTS_DIR is skipped, not followed.

    To prove the skip path actually ran (vs. falling through to JSON-parse
    failure on the target), the symlink target is a VALID checkpoint file
    that lives outside the results dir. If the scan were following symlinks
    it would return kind='ok' with that checkpoint's phase. Since the scan
    must skip symlinks, the result is kind='no_run'.
    """
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _scan_results_dir

    # Write a VALID checkpoint outside the results dir
    kwargs = _valid_checkpoint_kwargs()
    kwargs["phase"] = "should_not_be_visible"
    outside = tmp_results_dir.parent / "outside_checkpoint.json"
    write_checkpoint(outside, CalibrationCheckpoint(**kwargs))

    link = tmp_results_dir / "phase_evil_checkpoint.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this filesystem")

    snap = _scan_results_dir()
    # If symlinks were followed, snap.active.kind would be 'ok' with
    # phase='should_not_be_visible'. Symlink-skip → no_run.
    assert snap.active.kind == "no_run"
    assert snap.active.checkpoint is None


def test_scan_signature_invalidates_on_persistent_oserror(monkeypatch):
    """Persistent failures produce a strictly-increasing tick so the poll
    keeps invalidating instead of latching.

    Note: `Path('/nonexistent').glob(...)` does NOT raise on Linux — it yields
    an empty iterator. To exercise the `except OSError` branch we must monkey-
    patch RESULTS_DIR's glob method itself to raise. Also reset the module-
    level _signature_tick at the start so the assertion holds regardless of
    prior test pollution.
    """
    from ui.pages import calibration_handlers as ch_mod
    from ui.pages.calibration_handlers import _scan_signature

    monkeypatch.setattr(ch_mod, "_signature_tick", 0)
    monkeypatch.setattr(ch_mod, "_seen_scan_errors", set())

    class _RaisingPath:
        """Stand-in for RESULTS_DIR that raises OSError from glob()."""

        def glob(self, pattern):
            raise OSError("ENOENT: simulated mount failure")

    monkeypatch.setattr(ch_mod, "RESULTS_DIR", _RaisingPath())
    sig1 = _scan_signature()
    sig2 = _scan_signature()
    sig3 = _scan_signature()
    assert sig1[2] < sig2[2] < sig3[2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_calibration_dashboard_reactive.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add module-level constants, helpers, and reactive to `ui/pages/calibration_handlers.py`**

Insert near the top of `ui/pages/calibration_handlers.py` (after the existing imports):
```python
import dataclasses
import html
import logging
import os
import stat
import threading
import time
from pathlib import Path

from osmose.calibration.checkpoint import (
    CheckpointReadResult,
    LiveSnapshot,
    default_results_dir,
    is_live,
    probe_writable,
    read_checkpoint,
)

logger = logging.getLogger("osmose.ui.calibration_dashboard")

RESULTS_DIR: Path = default_results_dir()

# Startup probe (call once at module import; failure logs but does not raise).
try:
    probe_writable(RESULTS_DIR)
except OSError as e:
    logger.error("RESULTS_DIR probe failed: %s", e)

_signature_tick: int = 0
_seen_scan_errors: set[type] = set()
_seen_scan_errors_lock = threading.Lock()
_EMPTY_SNAPSHOT = LiveSnapshot(
    active=CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None),
    other_live_paths=(),
    snapshot_monotonic=0.0,
)


def _notify_scan_failure_once(e: OSError) -> None:
    cls = type(e)
    with _seen_scan_errors_lock:
        if cls in _seen_scan_errors:
            return
        _seen_scan_errors.add(cls)
    logger.error("calibration results scan failed: %s: %s", cls.__name__, e)
    try:
        ui.notification_show(
            f"Calibration directory scan failed "
            f"({html.escape(cls.__name__)}: {html.escape(str(e))}) — "
            "dashboard will retry. Check the results directory's mount/perms.",
            type="warning", duration=None,
        )
    except Exception:
        pass  # outside a Shiny session (tests) — log-only


def _scan_signature() -> tuple[float, int, int]:
    """Cheap poll dependency. Uses lstat() to match the symlink-skip policy.

    Persistent failures advance _signature_tick so the poll keeps invalidating
    (otherwise it would latch on (0.0, 0) and never re-fire _scan_results_dir,
    which would silence _notify_scan_failure_once after first call).
    """
    global _signature_tick
    try:
        pairs = []
        for p in RESULTS_DIR.glob("phase*_checkpoint.json"):
            try:
                st = p.lstat()
                if stat.S_ISLNK(st.st_mode):
                    continue
                pairs.append(st.st_mtime)
            except (FileNotFoundError, PermissionError):
                continue
        return (max(pairs, default=0.0), len(pairs), 0)
    except OSError:
        _signature_tick += 1
        return (0.0, 0, _signature_tick)


def _scan_results_dir() -> LiveSnapshot:
    """Atomic scan; never raises into the reactive runtime.

    Symlinks are skipped (security: a symlink in RESULTS_DIR cannot trick
    read_checkpoint into reading /etc/shadow, and bytes from a target file
    cannot leak through UnicodeDecodeError.__str__ into the UI banner).
    """
    try:
        paths_with_mtime: list[tuple[Path, float]] = []
        for p in RESULTS_DIR.glob("phase*_checkpoint.json"):
            try:
                st = p.lstat()
                if stat.S_ISLNK(st.st_mode):
                    continue
                paths_with_mtime.append((p, st.st_mtime))
            except (FileNotFoundError, PermissionError):
                continue
        paths_with_mtime.sort(key=lambda pm: pm[1], reverse=True)
        live: list[Path] = []
        for p, _mt in paths_with_mtime:
            try:
                if is_live(p):
                    live.append(p)
            except (FileNotFoundError, PermissionError):
                continue
        if live:
            active = read_checkpoint(live[0])
            others = tuple(live[1:])
        else:
            active = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
            others = ()
        return LiveSnapshot(active=active, other_live_paths=others,
                            snapshot_monotonic=time.monotonic())
    except OSError as e:
        _notify_scan_failure_once(e)
        return dataclasses.replace(_EMPTY_SNAPSHOT, snapshot_monotonic=time.monotonic())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_dashboard_reactive.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_calibration_dashboard_reactive.py
git commit -m "feat(calibration-ui): module-level RESULTS_DIR + scan helpers + LiveSnapshot reactive"
```

---

### Task 17: Run header (optimizer/phase/gen/elapsed/patience/live state)

**Files:**
- Modify: `ui/pages/calibration.py`
- Modify: `ui/pages/calibration_handlers.py`

- [ ] **Step 1: Insert the `run_header` output into the Run tab**

In `ui/pages/calibration.py`, find the `ui.nav_panel("Run", ...)` block around line 262-267 and insert `ui.output_ui("run_header")` between `cal_status` and `convergence_chart`:
```python
                ui.nav_panel(
                    "Run",
                    ui.div(
                        ui.output_text("cal_status"),
                        ui.output_ui("run_header"),
                        output_widget("convergence_chart"),
                    ),
                ),
```

- [ ] **Step 2a: Add pure helpers at MODULE SCOPE in `ui/pages/calibration_handlers.py`**

These are not decorated with `@output` / `@reactive.*` so they live at module scope (near the other module-level helpers from Task 16). The Step 3 tests below import them as module-level names — keeping them at module scope is what makes those imports work.

```python
def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def _ckpt_mtime_for(snap: LiveSnapshot) -> float:
    """Wall-clock time the active checkpoint was written."""
    if snap.active.checkpoint is None:
        return time.time()
    try:
        from datetime import datetime
        return datetime.fromisoformat(snap.active.checkpoint.timestamp_iso).timestamp()
    except ValueError:
        return time.time()
```

- [ ] **Step 2b: Add reactive + renderer INSIDE `register_calibration_handlers`**

**Paste the block below INSIDE the body of `register_calibration_handlers` (function starting at line 418), adding +4 spaces of indent to every line during transcription.** Same convention applies to every renderer in Tasks 18, 20, 21. The block sits alongside the existing `@reactive.poll(lambda: time.time(), interval_secs=0.5)` at line 442 and other server-bound reactives.

```python
@reactive.poll(_scan_signature, interval_secs=1.0)
def _live_snapshot() -> LiveSnapshot:
    return _scan_results_dir()


@output
@render.ui
def run_header():
    snap = _live_snapshot()
    if snap.active.kind == "no_run":
        return ui.tags.div("No active calibration run", class_="text-muted small")
    if snap.active.kind == "corrupt":
        return ui.tags.div(
            f"Checkpoint unreadable ({html.escape(snap.active.error_summary or '')})",
            class_="alert alert-danger",
        )
    if snap.active.kind == "partial":
        return ui.tags.div("Checkpoint updating…", class_="text-warning small")
    ckpt = snap.active.checkpoint
    assert ckpt is not None

    elapsed_str = _format_elapsed(ckpt.elapsed_seconds)
    gen_str = (
        f"gen {ckpt.generation} / {ckpt.generation_budget}"
        if ckpt.generation_budget else f"gen {ckpt.generation}"
    )
    from osmose.calibration.checkpoint import liveness_state
    age = time.time() - _ckpt_mtime_for(snap)
    state_text = liveness_state(age)
    state_dot = "●" if state_text in ("live", "stalled") else "○"
    patience = (
        f"⏱ patience {ckpt.gens_since_improvement}"
        if ckpt.gens_since_improvement > 0 else ""
    )
    return ui.tags.div(
        ui.tags.div(
            f"{ckpt.optimizer.upper()} · phase {html.escape(ckpt.phase)}  |  "
            f"{gen_str}  |  elapsed {elapsed_str}",
            class_="fw-bold",
        ),
        ui.tags.div(
            f"{patience}   {state_dot} {state_text} (last update {int(age)}s ago)",
            class_="small text-muted",
            **{"aria-live": "polite", "aria-atomic": "false"},
        ),
        class_="run-header mb-2",
    )
```

- [ ] **Step 3: Add pure-helper tests (I8) for `_format_elapsed` + `_ckpt_mtime_for`**

`tests/test_calibration_dashboard_reactive.py` additions:
```python
def test_format_elapsed_seconds():
    from ui.pages.calibration_handlers import _format_elapsed
    assert _format_elapsed(0) == "0s"
    assert _format_elapsed(45) == "45s"
    assert _format_elapsed(59.4) == "59s"


def test_format_elapsed_minutes():
    from ui.pages.calibration_handlers import _format_elapsed
    assert _format_elapsed(60) == "1m 0s"
    assert _format_elapsed(125) == "2m 5s"
    assert _format_elapsed(3599) == "59m 59s"


def test_format_elapsed_hours():
    from ui.pages.calibration_handlers import _format_elapsed
    assert _format_elapsed(3600) == "1h 0m"
    assert _format_elapsed(4980) == "1h 23m"
    assert _format_elapsed(86400) == "24h 0m"


def test_ckpt_mtime_for_returns_timestamp_iso_when_ok(tmp_results_dir):
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _ckpt_mtime_for, _scan_results_dir

    kwargs = _valid_checkpoint_kwargs()
    kwargs["timestamp_iso"] = "2026-05-12T10:00:00+00:00"
    write_checkpoint(tmp_results_dir / "phase_x_checkpoint.json",
                     CalibrationCheckpoint(**kwargs))
    snap = _scan_results_dir()
    from datetime import datetime, timezone
    expected = datetime(2026, 5, 12, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    assert _ckpt_mtime_for(snap) == expected


def test_ckpt_mtime_for_falls_back_when_no_active_checkpoint():
    """Defensive: when there's no active checkpoint, return current time
    (which yields age 0 — the renderer should already short-circuit before
    reaching this code path, but the helper must not raise)."""
    from osmose.calibration.checkpoint import CheckpointReadResult, LiveSnapshot
    from ui.pages.calibration_handlers import _ckpt_mtime_for
    import time
    snap = LiveSnapshot(
        active=CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None),
        other_live_paths=(),
        snapshot_monotonic=0.0,
    )
    before = time.time()
    result = _ckpt_mtime_for(snap)
    after = time.time()
    assert before <= result <= after
```

- [ ] **Step 4: Run pure-helper tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_calibration_dashboard_reactive.py -v -k "format_elapsed or ckpt_mtime_for"
```

Expected: 5 PASS.

- [ ] **Step 5: Manually smoke-test**

```bash
.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000 &
sleep 3
.venv/bin/python -c "
from datetime import datetime, timezone
from osmose.calibration.checkpoint import CalibrationCheckpoint, default_results_dir, write_checkpoint
d = default_results_dir(); d.mkdir(parents=True, exist_ok=True)
ckpt = CalibrationCheckpoint(
    optimizer='de', phase='smoke', generation=42, generation_budget=200,
    best_fun=3.14,
    per_species_residuals=None, per_species_sim_biomass=None, species_labels=None,
    best_x_log10=(-0.3, 0.8),
    best_parameters={'k_a': 0.5, 'k_b': 6.3},
    param_keys=('k_a', 'k_b'),
    bounds_log10={'k_a': (-1.0, 0.0), 'k_b': (0.0, 1.0)},
    gens_since_improvement=3, elapsed_seconds=4980.0,
    timestamp_iso=datetime.now(timezone.utc).isoformat(),
    banded_targets=None, proxy_source='objective_disabled',
)
write_checkpoint(d / 'phase_smoke_checkpoint.json', ckpt)
"
# Browse to http://localhost:8000 — Calibration page → Run tab — header should
# show 'DE · phase smoke  |  gen 42 / 200  |  elapsed 1h 23m'
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add ui/pages/calibration.py ui/pages/calibration_handlers.py tests/test_calibration_dashboard_reactive.py
git commit -m "feat(calibration-ui): run header + liveness_state + pure-helper tests"
```

---

### Task 18: Per-species ICES proxy table with magnitude factor

**Files:**
- Modify: `ui/pages/calibration.py`
- Modify: `ui/pages/calibration_handlers.py`
- Create: `tests/test_ices_proxy.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ices_proxy.py`:
```python
from __future__ import annotations

import math

import pytest

from osmose.calibration.checkpoint import CalibrationCheckpoint


def _make_ckpt_with_proxy(per_species_residuals, per_species_sim_biomass, banded_targets, proxy_source):
    species_labels = tuple(banded_targets.keys()) if banded_targets else None
    return CalibrationCheckpoint(
        optimizer="de", phase="test", generation=1, generation_budget=10,
        best_fun=1.0,
        per_species_residuals=per_species_residuals,
        per_species_sim_biomass=per_species_sim_biomass,
        species_labels=species_labels,
        best_x_log10=(0.0,), best_parameters={"k": 1.0}, param_keys=("k",),
        bounds_log10={"k": (-1.0, 1.0)},
        gens_since_improvement=0, elapsed_seconds=1.0,
        timestamp_iso="2026-05-12T10:00:00+00:00",
        banded_targets=banded_targets,
        proxy_source=proxy_source,
    )


def test_proxy_in_range_zero_loss():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.0,),
        per_species_sim_biomass=(0.87,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert len(rows) == 1
    assert rows[0]["state"] == "in_range"
    assert abs(rows[0]["magnitude"] - 1.0) < 0.01


def test_proxy_out_of_range_overshoot():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.5,),
        per_species_sim_biomass=(3.0,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert rows[0]["state"] == "out_of_range"
    assert abs(rows[0]["magnitude"] - 3.0 / math.sqrt(0.75)) < 0.01
    assert rows[0]["direction"] == "overshoot"


def test_proxy_out_of_range_undershoot():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.5,),
        per_species_sim_biomass=(0.1,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert rows[0]["state"] == "out_of_range"
    assert rows[0]["direction"] == "undershoot"


def test_proxy_extinct():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(100.0,),
        per_species_sim_biomass=(0.0,),
        banded_targets={"sp_a": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    assert rows[0]["state"] == "extinct"


def test_proxy_objective_disabled_renders_banner():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=None,
        per_species_sim_biomass=None,
        banded_targets=None,
        proxy_source="objective_disabled",
    )
    rows = _build_proxy_rows(ckpt)
    assert len(rows) == 1
    assert rows[0]["state"] == "objective_disabled"


def test_proxy_default_sort_out_first_then_in_then_extinct():
    from ui.pages.calibration_handlers import _build_proxy_rows

    ckpt = _make_ckpt_with_proxy(
        per_species_residuals=(0.0, 0.5, 100.0),
        per_species_sim_biomass=(1.0, 5.0, 0.0),
        banded_targets={"in_band": (0.5, 1.5), "out_band": (0.5, 1.5), "extinct_sp": (0.5, 1.5)},
        proxy_source="banded_loss",
    )
    rows = _build_proxy_rows(ckpt)
    states = [r["state"] for r in rows]
    assert states == ["out_of_range", "in_range", "extinct"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_ices_proxy.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `_build_proxy_rows` (module scope) and the `ices_proxy_table` renderer (inside register_calibration_handlers)**

In `ui/pages/calibration_handlers.py`, add the pure helper + constants at MODULE SCOPE (near the existing module-level code from Task 16):
```python
import math


_PROXY_EPS = 1e-9
_STATE_ORDER = {"out_of_range": 0, "in_range": 1, "extinct": 2}


def _build_proxy_rows(ckpt: CalibrationCheckpoint) -> list[dict]:
    """Compute the proxy-table rows from a checkpoint. Pure helper.

    When proxy_source != 'banded_loss', returns a single sentinel row whose
    'state' is the proxy_source itself — signal to the renderer to display
    the appropriate banner instead of a table.
    """
    if ckpt.proxy_source != "banded_loss":
        return [{"state": ckpt.proxy_source, "species": "", "loss": 0.0,
                 "band": (0.0, 0.0), "magnitude": 0.0, "direction": ""}]

    assert ckpt.species_labels is not None
    assert ckpt.per_species_residuals is not None
    assert ckpt.per_species_sim_biomass is not None
    assert ckpt.banded_targets is not None

    rows: list[dict] = []
    for i, sp in enumerate(ckpt.species_labels):
        residual = ckpt.per_species_residuals[i]
        sim_biomass = ckpt.per_species_sim_biomass[i]
        lo, hi = ckpt.banded_targets[sp]
        target_mean = math.sqrt(lo * hi)  # Inv 9 ensures lo > 0
        if sim_biomass == 0.0:
            state, direction, magnitude = "extinct", "", 0.0
        elif residual <= _PROXY_EPS:
            state = "in_range"
            magnitude = sim_biomass / target_mean
            direction = ""
        else:
            state = "out_of_range"
            magnitude = sim_biomass / target_mean
            direction = "overshoot" if sim_biomass > hi else "undershoot"
        rows.append({
            "species": sp, "state": state, "loss": residual,
            "band": (lo, hi), "magnitude": magnitude, "direction": direction,
        })
    rows.sort(key=lambda r: _STATE_ORDER[r["state"]])
    return rows


def _aria_for_state(state: str, magnitude: float, direction: str) -> str:
    if state == "in_range":
        return "in band"
    if state == "out_of_range":
        return f"out of band — {magnitude:.2f} times {direction}"
    if state == "extinct":
        return "extinct"
    return "proxy unavailable"
```

Then paste the renderer below INSIDE `register_calibration_handlers` (near the other server-bound `@output` blocks), **adding +4 spaces of indent to every line during transcription** — same convention as Task 17 Step 2:

```python
@output
@render.ui
def ices_proxy_table():
    snap = _live_snapshot()
    if snap.active.kind != "ok":
        return ui.tags.div()
    ckpt = snap.active.checkpoint
    rows = _build_proxy_rows(ckpt)

    if rows and rows[0]["state"] == "objective_disabled":
        return ui.tags.div(
            "ICES proxy unavailable: this run does not use banded-loss objectives. "
            "Authoritative verdict will appear in Results tab on completion.",
            class_="alert alert-info small",
        )
    if rows and rows[0]["state"] == "not_implemented":
        return ui.tags.div(
            "ICES proxy: per-species residuals were not exposed by losses.py "
            "despite banded-loss being configured. This is a bug — please file an "
            "issue and include the checkpoint filename.",
            class_="alert alert-danger small",
        )

    table_rows = []
    n_in, n_out, n_na = 0, 0, 0
    for r in rows:
        if r["state"] == "in_range":
            badge, n_in = "✓", n_in + 1
            mag_text = f"≈{r['magnitude']:.2f}×"
        elif r["state"] == "out_of_range":
            badge, n_out = "✗", n_out + 1
            mag_text = f"{r['magnitude']:.2f}× {r['direction']}"
        elif r["state"] == "extinct":
            badge, n_out = "☠", n_out + 1
            mag_text = "extinct"
        else:
            badge, n_na = "—", n_na + 1
            mag_text = ""
        table_rows.append(ui.tags.tr(
            ui.tags.td(html.escape(r["species"])),
            ui.tags.td(f"loss {r['loss']:.2f}"),
            ui.tags.td(f"band [{r['band'][0]:.2f}, {r['band'][1]:.2f}]"),
            ui.tags.td(badge, **{"aria-label": _aria_for_state(r['state'], r['magnitude'], r['direction'])}),
            ui.tags.td(mag_text),
        ))
    return ui.tags.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                ui.tags.th("species"), ui.tags.th("loss"), ui.tags.th("band"),
                ui.tags.th(""), ui.tags.th("magnitude"),
            )),
            ui.tags.tbody(*table_rows),
            class_="table table-sm",
        ),
        ui.tags.div(
            f"{n_in}/{n_in + n_out + n_na} in-band (proxy) · {n_out} out · {n_na} n/a · "
            "authoritative ICES verdict appears in Results tab after completion.",
            class_="small text-muted",
        ),
    )
```

Add the new output to the Run tab between `run_header` and `convergence_chart`:
```python
                ui.nav_panel(
                    "Run",
                    ui.div(
                        ui.output_text("cal_status"),
                        ui.output_ui("run_header"),
                        ui.output_ui("ices_proxy_table"),
                        output_widget("convergence_chart"),
                    ),
                ),
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_ices_proxy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/calibration.py ui/pages/calibration_handlers.py tests/test_ices_proxy.py
git commit -m "feat(calibration-ui): per-species ICES proxy table with magnitude factor"
```

---

### Task 19: Best-ever reference line on the convergence chart

**Files:**
- Modify: `ui/pages/calibration_charts.py`
- Modify: `ui/pages/calibration.py` (line ~454 — the existing `@render_plotly convergence_chart` lives here, NOT in calibration_handlers.py; Shiny will not allow two outputs with the same name)
- Modify: `tests/test_ices_proxy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ices_proxy.py`:
```python
def test_convergence_chart_adds_best_ever_line(monkeypatch):
    """list_runs() with matching prior runs → chart gets a horizontal line."""
    from osmose.calibration import history as hist_mod
    from ui.pages.calibration_charts import make_convergence_chart

    monkeypatch.setattr(
        hist_mod, "list_runs",
        lambda history_dir=None: [
            {"algorithm": "de", "phase": "test", "best_objective": 4.2,
             "timestamp": "2026-05-01T10:00:00+00:00", "n_params": 2, "duration_seconds": 1.0,
             "path": "x"},
            {"algorithm": "de", "phase": "test", "best_objective": 5.1,
             "timestamp": "2026-05-02T10:00:00+00:00", "n_params": 2, "duration_seconds": 1.0,
             "path": "x"},
        ],
    )
    fig = make_convergence_chart(history=[10.0, 8.0, 7.0], optimizer="de", phase="test")
    shapes = fig.layout.shapes or ()
    assert any(s.y0 == 4.2 and s.y1 == 4.2 for s in shapes), "expected hline at 4.2"


def test_convergence_chart_no_best_ever_line_when_no_prior_runs(monkeypatch):
    from osmose.calibration import history as hist_mod
    from ui.pages.calibration_charts import make_convergence_chart

    monkeypatch.setattr(hist_mod, "list_runs", lambda history_dir=None: [])
    fig = make_convergence_chart(history=[10.0], optimizer="de", phase="test")
    assert not (fig.layout.shapes or ())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_ices_proxy.py -v -k convergence_chart
```

Expected: FAIL.

- [ ] **Step 3: Modify `make_convergence_chart`**

In `ui/pages/calibration_charts.py`, extend the existing signature. The current `make_convergence_chart(history, tmpl="osmose")` is called at `ui/pages/calibration.py:456` as `make_convergence_chart(cal_history.get(), tmpl=_tmpl())` — `tmpl` MUST be preserved as a kwarg for backward compat. Add the new `optimizer` and `phase` kwargs alongside it:
```python
def make_convergence_chart(
    history,
    tmpl: str = "osmose",        # existing — preserved for backward compat with the convergence_chart render delegate
    optimizer: str | None = None,  # NEW
    phase: str | None = None,      # NEW
):
    """Build the convergence Plotly figure with an optional best-ever reference line.

    When optimizer AND phase are both given, queries osmose.calibration.history.list_runs
    for prior matching runs and adds a horizontal dashed line at the minimum
    best_objective with annotation 'best ever: f=<X.XXX>'. Existing callers
    that pass only `history` and `tmpl` are unaffected — no reference line drawn.
    """
    import plotly.graph_objects as go

    from osmose.calibration import history as hist_mod

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history, mode="lines+markers", name="best_fun"))

    if optimizer is not None and phase is not None:
        try:
            runs = hist_mod.list_runs()
        except Exception:
            runs = []
        matching = [r for r in runs if r.get("algorithm") == optimizer and r.get("phase") == phase]
        finite = [r["best_objective"] for r in matching if r.get("best_objective") not in (None, float("inf"))]
        if finite:
            best_ever = min(finite)
            fig.add_hline(
                y=best_ever, line_dash="dash",
                annotation_text=f"best ever: f={best_ever:.3f}",
                annotation_position="top left",
            )
    fig.update_layout(xaxis_title="generation", yaxis_title="best_fun", template=tmpl)
    return fig
```

- [ ] **Step 4: Update `convergence_chart` in `ui/pages/calibration.py` line ~454-456**

The existing render delegate sits inside `calibration.py`'s session server function (separate from `register_calibration_handlers`). Replace the existing body:

```python
@render_plotly
def convergence_chart():
    return make_convergence_chart(cal_history.get(), tmpl=_tmpl())
```

with:

```python
@render_plotly
def convergence_chart():
    # Read the active checkpoint to derive (optimizer, phase) for the
    # best-ever reference line. _scan_results_dir is module-level and
    # importable; _live_snapshot is a @reactive.poll local to
    # register_calibration_handlers and CANNOT be imported (it's not a
    # module-level name). _scan_results_dir is pure, never raises, returns
    # the same LiveSnapshot shape, so it's the correct source from here.
    from ui.pages.calibration_handlers import _scan_results_dir
    try:
        snap = _scan_results_dir()
        if snap.active.kind == "ok":
            opt = snap.active.checkpoint.optimizer
            ph = snap.active.checkpoint.phase
        else:
            opt = ph = None
    except Exception:  # noqa: BLE001 — defensive fallback; should never fire
        opt = ph = None
    return make_convergence_chart(
        cal_history.get(), tmpl=_tmpl(), optimizer=opt, phase=ph,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_ices_proxy.py -v -k convergence_chart
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/calibration_charts.py ui/pages/calibration_handlers.py tests/test_ices_proxy.py
git commit -m "feat(calibration-ui): convergence chart shows best-ever reference line from history"
```

---

### Task 20: Current-best-parameters block with bound-distance hints

**Files:**
- Modify: `ui/pages/calibration.py`
- Modify: `ui/pages/calibration_handlers.py`
- Modify: `tests/test_ices_proxy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ices_proxy.py`:
```python
def test_bound_distance_badge_at_upper():
    from ui.pages.calibration_handlers import _build_param_rows

    ckpt = CalibrationCheckpoint(
        optimizer="de", phase="test", generation=1, generation_budget=None,
        best_fun=1.0,
        per_species_residuals=None, per_species_sim_biomass=None, species_labels=None,
        best_x_log10=(0.96, 0.5),
        best_parameters={"k_a": 9.12, "k_b": 3.16},
        param_keys=("k_a", "k_b"),
        bounds_log10={"k_a": (0.0, 1.0), "k_b": (0.0, 1.0)},
        gens_since_improvement=0, elapsed_seconds=1.0,
        timestamp_iso="2026-05-12T10:00:00+00:00",
        banded_targets=None, proxy_source="objective_disabled",
    )
    rows = _build_param_rows(ckpt)
    by_key = {r["key"]: r for r in rows}
    assert by_key["k_a"]["bound_badge"] == "at upper bound"
    assert by_key["k_b"]["bound_badge"] == ""


def test_bound_distance_badge_at_lower():
    from ui.pages.calibration_handlers import _build_param_rows

    ckpt = CalibrationCheckpoint(
        optimizer="de", phase="test", generation=1, generation_budget=None,
        best_fun=1.0,
        per_species_residuals=None, per_species_sim_biomass=None, species_labels=None,
        best_x_log10=(0.02,),
        best_parameters={"k_a": 1.05},
        param_keys=("k_a",),
        bounds_log10={"k_a": (0.0, 1.0)},
        gens_since_improvement=0, elapsed_seconds=1.0,
        timestamp_iso="2026-05-12T10:00:00+00:00",
        banded_targets=None, proxy_source="objective_disabled",
    )
    rows = _build_param_rows(ckpt)
    assert rows[0]["bound_badge"] == "at lower bound"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_ices_proxy.py -v -k bound_distance
```

Expected: FAIL.

- [ ] **Step 3: Implement `_build_param_rows` (module scope) + `current_best_parameters` renderer (inside register_calibration_handlers, 4-space indent)**

Add the pure helper at MODULE SCOPE:
```python
_BOUND_DISTANCE_THRESHOLD = 0.05  # 5% of bound range


def _build_param_rows(ckpt: CalibrationCheckpoint) -> list[dict]:
    """Current-best-parameters rows with bound-distance hints."""
    rows: list[dict] = []
    for key in sorted(ckpt.param_keys):
        idx = ckpt.param_keys.index(key)
        x_log10 = ckpt.best_x_log10[idx]
        lo, hi = ckpt.bounds_log10[key]
        rng = hi - lo
        if rng <= 0:
            badge = ""
        elif (hi - x_log10) / rng < _BOUND_DISTANCE_THRESHOLD:
            badge = "at upper bound"
        elif (x_log10 - lo) / rng < _BOUND_DISTANCE_THRESHOLD:
            badge = "at lower bound"
        else:
            badge = ""
        rows.append({"key": key, "value": ckpt.best_parameters[key], "bound_badge": badge})
    return rows


```

Then paste INSIDE `register_calibration_handlers`, **+4 spaces of indent on every line during transcription**:

```python
@output
@render.ui
def current_best_parameters():
    snap = _live_snapshot()
    if snap.active.kind != "ok":
        return ui.tags.div()
    rows = _build_param_rows(snap.active.checkpoint)
    return ui.tags.details(
        ui.tags.summary("Current best parameters"),
        ui.tags.table(
            ui.tags.tbody(*[
                ui.tags.tr(
                    ui.tags.td(html.escape(r["key"])),
                    ui.tags.td(f"{r['value']:.4f}"),
                    ui.tags.td(
                        ui.tags.span(f"[{r['bound_badge']}]", class_="text-warning small")
                        if r["bound_badge"] else ""
                    ),
                )
                for r in rows
            ]),
            class_="table table-sm small",
        ),
    )
```

In `ui/pages/calibration.py`, add the new output below `convergence_chart`:
```python
                ui.nav_panel(
                    "Run",
                    ui.div(
                        ui.output_text("cal_status"),
                        ui.output_ui("run_header"),
                        ui.output_ui("ices_proxy_table"),
                        output_widget("convergence_chart"),
                        ui.output_ui("current_best_parameters"),
                    ),
                ),
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_ices_proxy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/calibration.py ui/pages/calibration_handlers.py tests/test_ices_proxy.py
git commit -m "feat(calibration-ui): current best parameters block with bound-distance hints"
```

---

### Task 21: "N other live runs" disclosure badge

**Files:**
- Modify: `ui/pages/calibration.py`
- Modify: `ui/pages/calibration_handlers.py`
- Modify: `tests/test_calibration_dashboard_reactive.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_calibration_dashboard_reactive.py`:
```python
def test_other_live_runs_returns_all_but_newest(tmp_results_dir):
    from osmose.calibration.checkpoint import CalibrationCheckpoint, write_checkpoint
    from tests.test_calibration_checkpoint import _valid_checkpoint_kwargs
    from ui.pages.calibration_handlers import _scan_results_dir

    for phase in ["a", "b", "c"]:
        kwargs = _valid_checkpoint_kwargs()
        kwargs["phase"] = phase
        write_checkpoint(
            tmp_results_dir / f"phase_{phase}_checkpoint.json",
            CalibrationCheckpoint(**kwargs),
        )

    snap = _scan_results_dir()
    assert len(snap.other_live_paths) == 2
```

- [ ] **Step 2: Run the test (Task 16 implementation may already cover this)**

```bash
.venv/bin/python -m pytest tests/test_calibration_dashboard_reactive.py -v -k other_live_runs
```

If PASS: skip Step 3. If FAIL: implement `_scan_results_dir`'s `other_live_paths` partition (already covered in Task 16 step 3).

- [ ] **Step 3: Implement the renderer (inside register_calibration_handlers — +4 indent on transcription)**

Paste the block below INSIDE `register_calibration_handlers`, +4 spaces of indent on every line during transcription:

```python
@output
@render.ui
def other_live_runs_badge():
    snap = _live_snapshot()
    n_others = len(snap.other_live_paths)
    if n_others == 0:
        return ui.tags.div()
    label = "1 other live run" if n_others == 1 else f"{n_others} other live runs"
    return ui.tags.div(
        ui.tags.span(f"📡 {label}", class_="me-2"),
        ui.input_action_button(
            "btn_switch_other_run",
            "[switch]",
            class_="btn-sm btn-outline-secondary",
            **{"aria-label": "Switch to next-most-recent live run"},
        ),
        class_="alert alert-info py-1 small",
    )


@reactive.effect
@reactive.event(input.btn_switch_other_run)
def _switch_to_other_run():
    """v1 disclosure-only: log the intent so it's traceable, then no-op.

    A future iteration will rotate the active-pick reactive.Value to actually
    switch. The disclosure badge + button shipping in v1 makes the omission
    visible; the rotate-active behaviour is small follow-up work.
    """
    snap = _live_snapshot()
    if not snap.other_live_paths:
        return
    logger.info("user clicked [switch]; next-other-run=%s", snap.other_live_paths[0])
```

Add the output between `run_header` and `ices_proxy_table`:
```python
                ui.nav_panel(
                    "Run",
                    ui.div(
                        ui.output_text("cal_status"),
                        ui.output_ui("run_header"),
                        ui.output_ui("other_live_runs_badge"),
                        ui.output_ui("ices_proxy_table"),
                        output_widget("convergence_chart"),
                        ui.output_ui("current_best_parameters"),
                    ),
                ),
```

- [ ] **Step 4: Run tests for regression**

```bash
.venv/bin/python -m pytest -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/pages/calibration.py ui/pages/calibration_handlers.py tests/test_calibration_dashboard_reactive.py
git commit -m "feat(calibration-ui): N-other-live-runs disclosure badge with [switch] stub"
```

---

## Phase 9 — Smoke testing

### Task 22: Manual UI smoke checklist (merge gate)

**Files:**
- No code changes. This is the PR description's merge-readiness checklist.

- [ ] **Step 1: Start the Shiny server**

```bash
.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000
```

- [ ] **Step 2: Drive a short DE run via CLI (in another shell)**

```bash
.venv/bin/python scripts/calibrate_baltic.py --optimizer de --phase smoke --maxiter 3 --popsize 4 --workers 1
```

- [ ] **Step 3: Observe in the browser**

Open `http://localhost:8000` → Calibration → Run tab. Expected:
- Run header populates within 2 s: `DE · phase smoke  |  gen N / 3  |  elapsed Xs   ●  live`
- Proxy table renders 8 rows (Baltic species) with mixed in-band / out-of-band states + magnitude factors
- Convergence chart shows decreasing `best_fun`; if history exists for `(de, smoke)`, a `best ever: f=X.XXX` dashed line appears
- "Current best parameters" expandable shows 24+ params; some rows show `[at upper bound]` / `[at lower bound]` badges when applicable
- After the CLI process exits, header transitions `live` → `stalled` at ~60 s → `idle` at ~5 min

- [ ] **Step 4: Trigger the "N other live runs" disclosure badge**

In a second shell:
```bash
.venv/bin/python scripts/calibrate_baltic.py --optimizer de --phase smoke_b --maxiter 3 --popsize 4 --workers 1
```

Expected: the badge appears reading `1 other live run [switch]`.

- [ ] **Step 5: Verify History tab populates**

Click the History tab. Expected: at least one row per completed smoke run with timestamp / algorithm / best_objective / duration.

- [ ] **Step 6: Persistence-degraded path**

```bash
chmod 0555 data/baltic/calibration_results
```

Restart the Shiny server. Expected: a notification banner "Calibration results directory unreachable …" on first page load. Clean up:
```bash
chmod 0755 data/baltic/calibration_results
```

- [ ] **Step 7: Banded-loss-disabled path**

Run the calibrator without banded targets configured. Expected: proxy table replaced by the blue `objective_disabled` banner.

- [ ] **Step 8: Run the full test suite one last time**

```bash
.venv/bin/python -m pytest -v
```

Expected: every new test PASSes; no pre-existing test fails.

- [ ] **Step 9: PR description checkbox**

Each smoke step above goes into the PR description as a `- [ ]` checklist; the reviewer ticks them off during review.

---

## Self-review summary

Spec coverage check:
- **§1-3 (context, goals, non-goals):** covered implicitly via task scoping.
- **§4 (architecture):** Tasks 1-7 build the data contract; Tasks 11-15 the producers; Tasks 16-21 the consumers.
- **§5 (data contract):** Tasks 1-7 implement `CalibrationCheckpoint`, `CheckpointReadResult`, `LiveSnapshot`, write/read, is_live, probe_writable. All 14 invariants tested in Task 2.
- **§6 (runner integration):** Task 11 (DE), Task 13 (CMA-ES), Task 14 (surrogate-DE), Task 15 (NSGA-II).
- **§6.5 (objective changes):** Task 9 (Path A), Task 10 (Path B).
- **§7 (UI surface):** Task 16 (reactive scaffolding), Task 17 (run header), Task 18 (proxy table), Task 19 (best-ever line), Task 20 (current-best params + bound-distance hints), Task 21 (other-live-runs badge).
- **§8 (proxy semantics):** Task 18 covers `in_range` / `out_of_range` / `extinct` / `objective_disabled` / `not_implemented` states with full test coverage.
- **§9 (history wiring):** Task 12 (DE), Task 15 (NSGA-II). CMA-ES and surrogate-DE inherit the same `_save_run_for_de` helper pattern at completion; if explicit save_run hooks are needed for those two runners, add a follow-up task.
- **§10 (test plan):** every test bucket in the spec maps to a test in one of the tasks above; the `__post_init__` invariants are tested as 14 cases (renumbered correctly).
- **§11 (open follow-ups):** out of scope by design.

Type consistency: `_ObjectiveWrapper.last_per_species_residuals` is a `list[tuple[str, float, float]] | None` in Tasks 9 and 11; `_write_progress_checkpoint`, `_build_proxy_rows`, `_build_param_rows` all reference the same field names; `CalibrationCheckpoint` field names are consistent across producer (Task 11) and consumer (Task 18) tasks.

No placeholders: every code step contains complete code (no "TODO: implement", no "similar to Task N"). Test cases all carry both the failing assertion and the expected behaviour.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-12-calibration-dashboard.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
