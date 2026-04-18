# Calibration Library Gaps — Phase 1 Implementation Plan

> **STATUS (verified 2026-04-18): COMPLETE — shipped alongside Calibration Library v1 (2026-04-15). DO NOT RE-EXECUTE.** Evidence: `osmose/calibration/targets.py`, `osmose/calibration/losses.py`, `osmose/calibration/multiseed.py` all exist with expected public symbols (`BiomassTarget`, `banded_log_ratio_loss`, `validate_multiseed`); `__init__.py` exports confirmed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract reusable calibration primitives from the Baltic script into the `osmose/calibration/` library: ICES target data model, composable banded loss objectives, evaluation caching, schema validation, surrogate cross-validation, multi-objective sensitivity, and multi-seed validation.

**Architecture:** Six independent modules with minimal cross-dependencies. `targets.py` is imported by `losses.py`; all others are standalone. New modules follow existing patterns in `osmose/calibration/` (pure functions + dataclasses, no framework dependencies). TDD throughout — test first, implement, commit.

**Tech Stack:** Python 3.12+, numpy, scipy, scikit-learn (existing), SALib (existing), pymoo (existing), pytest

**Spec:** `docs/superpowers/specs/2026-04-15-calibration-library-gaps-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `osmose/calibration/targets.py` | **create** | `BiomassTarget` dataclass, `load_targets()` with `#!` metadata parsing |
| `osmose/calibration/losses.py` | **create** | `banded_log_ratio_loss()`, `stability_penalty()`, `worst_species_penalty()`, `make_banded_objective()` |
| `osmose/calibration/multiseed.py` | **create** | `validate_multiseed()`, `rank_candidates_multiseed()` |
| `osmose/calibration/problem.py` | **modify** | Add `enable_cache`, `cache_dir`, `registry` params; `_cache_key()`, `cache_stats()`, `clear_cache()`, `_validate_overrides()` |
| `osmose/calibration/surrogate.py` | **modify** | Add `fit_score_` attribute, `cross_validate()` method |
| `osmose/calibration/sensitivity.py` | **modify** | Extend `analyze()` for 2D Y arrays |
| `osmose/calibration/__init__.py` | **modify** | Export new public API |
| `data/baltic/reference/biomass_targets.csv` | **modify** | Add `reference_point_type` column and `#!` metadata lines |
| `tests/test_calibration_targets.py` | **create** | Tests for targets module |
| `tests/test_calibration_losses.py` | **create** | Tests for losses module |
| `tests/test_calibration_multiseed.py` | **create** | Tests for multiseed module |
| `tests/test_calibration_problem.py` | **modify** | Add cache + validation tests |
| `tests/test_calibration_surrogate.py` | **create** | Tests for cross-validation + fit_score_ |
| `tests/test_calibration_sensitivity.py` | **create** | Tests for 2D sensitivity |

---

### Task 1: `targets.py` — BiomassTarget dataclass and load_targets()

**Files:**
- Create: `osmose/calibration/targets.py`
- Create: `tests/test_calibration_targets.py`

- [ ] **Step 1: Write tests for BiomassTarget and load_targets()**

```python
# tests/test_calibration_targets.py
"""Tests for BiomassTarget data model and CSV loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from osmose.calibration.targets import BiomassTarget, load_targets


class TestBiomassTarget:
    def test_required_fields(self) -> None:
        t = BiomassTarget(species="cod", target=120000, lower=60000, upper=250000)
        assert t.species == "cod"
        assert t.target == 120000
        assert t.lower == 60000
        assert t.upper == 250000

    def test_defaults(self) -> None:
        t = BiomassTarget(species="cod", target=1, lower=0, upper=2)
        assert t.weight == 1.0
        assert t.reference_point_type == "biomass"
        assert t.source == ""
        assert t.notes == ""

    def test_all_fields(self) -> None:
        t = BiomassTarget(
            species="cod", target=120000, lower=60000, upper=250000,
            weight=0.5, reference_point_type="ssb", source="ICES", notes="test",
        )
        assert t.reference_point_type == "ssb"
        assert t.source == "ICES"


class TestLoadTargets:
    def test_basic_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "targets.csv"
        csv.write_text(textwrap.dedent("""\
            species,target_tonnes,lower_tonnes,upper_tonnes,weight,reference_point_type,source,notes
            cod,120000,60000,250000,1.0,ssb,ICES,collapse
            herring,1500000,800000,3000000,1.0,biomass,ICES,complex
        """))
        targets, metadata = load_targets(csv)
        assert len(targets) == 2
        assert targets[0].species == "cod"
        assert targets[0].target == 120000
        assert targets[0].reference_point_type == "ssb"
        assert targets[1].species == "herring"
        assert metadata == {}

    def test_backward_compat_no_reference_point_type(self, tmp_path: Path) -> None:
        """Old-format CSV without reference_point_type column defaults to 'biomass'."""
        csv = tmp_path / "targets.csv"
        csv.write_text(textwrap.dedent("""\
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """))
        targets, metadata = load_targets(csv)
        assert len(targets) == 1
        assert targets[0].reference_point_type == "biomass"
        assert targets[0].source == ""
        assert targets[0].notes == ""

    def test_comment_lines_skipped(self, tmp_path: Path) -> None:
        csv = tmp_path / "targets.csv"
        csv.write_text(textwrap.dedent("""\
            # This is a comment
            # Another comment
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """))
        targets, _ = load_targets(csv)
        assert len(targets) == 1

    def test_metadata_lines(self, tmp_path: Path) -> None:
        csv = tmp_path / "targets.csv"
        csv.write_text(textwrap.dedent("""\
            #! version: 1.0
            #! last_updated: 2026-04-15
            # Human comment
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """))
        targets, metadata = load_targets(csv)
        assert len(targets) == 1
        assert metadata["version"] == "1.0"
        assert metadata["last_updated"] == "2026-04-15"

    def test_malformed_metadata_line_ignored(self, tmp_path: Path) -> None:
        """#! line without a colon is silently ignored."""
        csv = tmp_path / "targets.csv"
        csv.write_text(textwrap.dedent("""\
            #! no-colon-here
            species,target_tonnes,lower_tonnes,upper_tonnes,weight
            cod,120000,60000,250000,1.0
        """))
        targets, metadata = load_targets(csv)
        assert len(targets) == 1
        assert metadata == {}

    def test_loads_real_baltic_csv(self) -> None:
        """Smoke test: loads the actual data file."""
        csv = Path("data/baltic/reference/biomass_targets.csv")
        if not csv.exists():
            pytest.skip("Baltic targets CSV not found")
        targets, _ = load_targets(csv)
        assert len(targets) >= 8
        species = [t.species for t in targets]
        assert "cod" in species
        assert "herring" in species
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_targets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.targets'`

- [ ] **Step 3: Implement targets.py**

```python
# osmose/calibration/targets.py
"""ICES biomass target data model and CSV loader."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BiomassTarget:
    """A calibration target for one species, typically from ICES stock assessments."""

    species: str
    target: float
    lower: float
    upper: float
    weight: float = 1.0
    reference_point_type: str = "biomass"
    source: str = ""
    notes: str = ""


def load_targets(path: Path) -> tuple[list[BiomassTarget], dict]:
    """Load calibration targets from CSV.

    Skips ``#`` comment lines.  Lines starting with ``#!`` are parsed as
    key-value metadata (``str.split(":", 1)``).  The ``reference_point_type``,
    ``source``, and ``notes`` columns are optional for backward compatibility.

    Returns:
        (targets, metadata) tuple.
    """
    metadata: dict[str, str] = {}
    data_lines: list[str] = []

    text = path.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            payload = stripped[2:].strip()
            if ":" in payload:
                key, value = payload.split(":", 1)
                metadata[key.strip()] = value.strip()
        elif stripped.startswith("#"):
            continue
        elif stripped:
            data_lines.append(stripped)

    targets: list[BiomassTarget] = []
    if not data_lines:
        return targets, metadata

    reader = csv.DictReader(data_lines)
    for row in reader:
        targets.append(
            BiomassTarget(
                species=row["species"],
                target=float(row["target_tonnes"]),
                lower=float(row["lower_tonnes"]),
                upper=float(row["upper_tonnes"]),
                weight=float(row.get("weight", "1.0")),
                reference_point_type=row.get("reference_point_type", "biomass"),
                source=row.get("source", ""),
                notes=row.get("notes", ""),
            )
        )

    return targets, metadata
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_targets.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```
git add osmose/calibration/targets.py tests/test_calibration_targets.py
git commit -m "feat(calibration): add BiomassTarget data model and CSV loader"
```

---

### Task 2: `losses.py` — Composable banded loss objectives

**Files:**
- Create: `osmose/calibration/losses.py`
- Create: `tests/test_calibration_losses.py`

- [ ] **Step 1: Write tests for loss primitives and factory**

```python
# tests/test_calibration_losses.py
"""Tests for composable banded loss objectives."""

from __future__ import annotations

import math

import pytest

from osmose.calibration.losses import (
    banded_log_ratio_loss,
    make_banded_objective,
    stability_penalty,
    worst_species_penalty,
)
from osmose.calibration.targets import BiomassTarget


class TestBandedLogRatioLoss:
    def test_within_range_returns_zero(self) -> None:
        assert banded_log_ratio_loss(150000, 60000, 250000) == 0.0

    def test_at_lower_bound_returns_zero(self) -> None:
        assert banded_log_ratio_loss(60000, 60000, 250000) == 0.0

    def test_at_upper_bound_returns_zero(self) -> None:
        assert banded_log_ratio_loss(250000, 60000, 250000) == 0.0

    def test_below_range(self) -> None:
        result = banded_log_ratio_loss(6000, 60000, 250000)
        expected = math.log10(60000 / 6000) ** 2
        assert result == pytest.approx(expected)

    def test_above_range(self) -> None:
        result = banded_log_ratio_loss(500000, 60000, 250000)
        expected = math.log10(500000 / 250000) ** 2
        assert result == pytest.approx(expected)

    def test_zero_biomass(self) -> None:
        assert banded_log_ratio_loss(0, 60000, 250000) == 100.0

    def test_negative_biomass(self) -> None:
        assert banded_log_ratio_loss(-100, 60000, 250000) == 100.0


class TestStabilityPenalty:
    def test_both_below_threshold(self) -> None:
        assert stability_penalty(0.1, 0.02) == 0.0

    def test_cv_above_threshold(self) -> None:
        result = stability_penalty(0.4, 0.02)
        assert result == pytest.approx((0.4 - 0.2) ** 2)

    def test_trend_above_threshold(self) -> None:
        result = stability_penalty(0.1, 0.1)
        assert result == pytest.approx((0.1 - 0.05) ** 2)

    def test_both_above_threshold(self) -> None:
        result = stability_penalty(0.5, 0.15)
        expected = (0.5 - 0.2) ** 2 + (0.15 - 0.05) ** 2
        assert result == pytest.approx(expected)

    def test_custom_thresholds(self) -> None:
        result = stability_penalty(0.6, 0.3, cv_threshold=0.5, trend_threshold=0.2)
        expected = (0.6 - 0.5) ** 2 + (0.3 - 0.2) ** 2
        assert result == pytest.approx(expected)


class TestWorstSpeciesPenalty:
    def test_returns_max(self) -> None:
        assert worst_species_penalty([0.1, 0.5, 0.3]) == 0.5

    def test_single_species(self) -> None:
        assert worst_species_penalty([0.7]) == 0.7

    def test_all_zero(self) -> None:
        assert worst_species_penalty([0.0, 0.0]) == 0.0


class TestMakeBandedObjective:
    @pytest.fixture()
    def targets(self) -> list[BiomassTarget]:
        return [
            BiomassTarget(species="cod", target=120000, lower=60000, upper=250000, weight=1.0),
            BiomassTarget(species="herring", target=1500000, lower=800000, upper=3000000, weight=0.5),
        ]

    def test_all_within_range(self, targets: list[BiomassTarget]) -> None:
        obj = make_banded_objective(targets, ["cod", "herring"])
        stats = {
            "cod_mean": 150000, "cod_cv": 0.1, "cod_trend": 0.01,
            "herring_mean": 2000000, "herring_cv": 0.05, "herring_trend": 0.02,
        }
        assert obj(stats) == 0.0

    def test_one_below_range(self, targets: list[BiomassTarget]) -> None:
        obj = make_banded_objective(targets, ["cod", "herring"])
        stats = {
            "cod_mean": 10000, "cod_cv": 0.1, "cod_trend": 0.01,
            "herring_mean": 2000000, "herring_cv": 0.05, "herring_trend": 0.02,
        }
        result = obj(stats)
        assert result > 0.0

    def test_missing_species_key_penalty(self, targets: list[BiomassTarget]) -> None:
        obj = make_banded_objective(targets, ["cod", "herring"])
        stats = {"cod_mean": 150000, "cod_cv": 0.1, "cod_trend": 0.01}
        # herring keys missing -> 100.0 penalty * 0.5 weight = 50.0 + worst term
        result = obj(stats)
        assert result >= 50.0

    def test_stability_penalty_applied(self, targets: list[BiomassTarget]) -> None:
        obj = make_banded_objective(targets, ["cod", "herring"], w_stability=5.0)
        stats = {
            "cod_mean": 150000, "cod_cv": 0.5, "cod_trend": 0.01,
            "herring_mean": 2000000, "herring_cv": 0.05, "herring_trend": 0.02,
        }
        result = obj(stats)
        # Cod CV=0.5 > 0.2 threshold -> penalty
        assert result > 0.0

    def test_worst_species_term(self, targets: list[BiomassTarget]) -> None:
        obj_with = make_banded_objective(targets, ["cod", "herring"], w_worst=1.0)
        obj_without = make_banded_objective(targets, ["cod", "herring"], w_worst=0.0)
        stats = {
            "cod_mean": 10000, "cod_cv": 0.1, "cod_trend": 0.01,
            "herring_mean": 2000000, "herring_cv": 0.05, "herring_trend": 0.02,
        }
        assert obj_with(stats) > obj_without(stats)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_losses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.calibration.losses'`

- [ ] **Step 3: Implement losses.py**

```python
# osmose/calibration/losses.py
"""Composable banded loss objectives for OSMOSE calibration."""

from __future__ import annotations

import math
from collections.abc import Callable

from osmose.calibration.targets import BiomassTarget


def banded_log_ratio_loss(sim_biomass: float, lower: float, upper: float) -> float:
    """Per-species loss: 0 inside [lower, upper], squared log-distance outside."""
    if sim_biomass <= 0:
        return 100.0
    if sim_biomass < lower:
        return math.log10(lower / sim_biomass) ** 2
    if sim_biomass > upper:
        return math.log10(sim_biomass / upper) ** 2
    return 0.0


def stability_penalty(
    cv: float,
    trend: float,
    cv_threshold: float = 0.2,
    trend_threshold: float = 0.05,
) -> float:
    """Penalty for oscillations (CV) and non-equilibrium (trend)."""
    penalty = 0.0
    if cv > cv_threshold:
        penalty += (cv - cv_threshold) ** 2
    if trend > trend_threshold:
        penalty += (trend - trend_threshold) ** 2
    return penalty


def worst_species_penalty(species_errors: list[float]) -> float:
    """Max of weighted per-species errors."""
    return max(species_errors)


def make_banded_objective(
    targets: list[BiomassTarget],
    species_names: list[str],
    w_stability: float = 5.0,
    w_worst: float = 0.5,
) -> Callable[[dict[str, float]], float]:
    """Factory: returns callable(species_stats) -> scalar objective.

    species_stats keys: ``{species}_mean``, ``{species}_cv``, ``{species}_trend``.
    Missing species keys receive a penalty of 100.0, weighted by species weight.
    (Note: the Baltic script applies the missing-species penalty unweighted.
    Weighting it here is intentional — see spec for rationale.)
    """
    target_dict = {t.species: t for t in targets}

    def objective(species_stats: dict[str, float]) -> float:
        total_error = 0.0
        weighted_errors: list[float] = []

        for sp in species_names:
            mean_key = f"{sp}_mean"
            cv_key = f"{sp}_cv"
            trend_key = f"{sp}_trend"

            if mean_key not in species_stats:
                sp_error = 100.0
            else:
                sp_error = banded_log_ratio_loss(
                    species_stats[mean_key], target_dict[sp].lower, target_dict[sp].upper
                )

            w = target_dict[sp].weight
            weighted_error = w * sp_error
            total_error += weighted_error
            weighted_errors.append(weighted_error)

            cv = species_stats.get(cv_key, 0.0)
            trend = species_stats.get(trend_key, 0.0)
            total_error += w_stability * w * stability_penalty(cv, trend)

        total_error += w_worst * worst_species_penalty(weighted_errors)
        return total_error

    return objective
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_losses.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```
git add osmose/calibration/losses.py tests/test_calibration_losses.py
git commit -m "feat(calibration): add composable banded loss objectives"
```

---

### Task 3: `multiseed.py` — Multi-seed validation

**Files:**
- Create: `osmose/calibration/multiseed.py`
- Create: `tests/test_calibration_multiseed.py`

- [ ] **Step 1: Write tests for validate_multiseed and rank_candidates_multiseed**

```python
# tests/test_calibration_multiseed.py
"""Tests for multi-seed validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.multiseed import rank_candidates_multiseed, validate_multiseed


def _make_mock_factory(base_value: float = 1.0, seed_noise: float = 0.1):
    """Returns a factory(seed) -> objective(x) -> float."""

    def factory(seed: int):
        rng = np.random.default_rng(seed)

        def objective(x: np.ndarray) -> float:
            return base_value + rng.normal(0, seed_noise) + float(np.sum(x))

        return objective

    return factory


class TestValidateMultiseed:
    def test_returns_expected_keys(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[1, 2, 3])
        assert "per_seed" in result
        assert "mean" in result
        assert "std" in result
        assert "cv" in result
        assert "worst_seed" in result
        assert "worst_value" in result

    def test_per_seed_length_matches_seeds(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[10, 20, 30, 40])
        assert len(result["per_seed"]) == 4

    def test_mean_is_average_of_per_seed(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[1, 2])
        assert result["mean"] == pytest.approx(np.mean(result["per_seed"]))

    def test_worst_seed_is_max(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[1, 2, 3])
        worst_idx = np.argmax(result["per_seed"])
        assert result["worst_seed"] == [1, 2, 3][worst_idx]
        assert result["worst_value"] == max(result["per_seed"])

    def test_deterministic(self) -> None:
        factory = _make_mock_factory()
        x = np.array([0.5])
        r1 = validate_multiseed(factory, x, seeds=[42, 123])
        r2 = validate_multiseed(factory, x, seeds=[42, 123])
        assert r1["per_seed"] == r2["per_seed"]


class TestRankCandidatesMultiseed:
    def test_returns_expected_keys(self) -> None:
        factory = _make_mock_factory()
        candidates = np.array([[0.0], [1.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2])
        assert "rankings" in result
        assert "scores" in result

    def test_rankings_length(self) -> None:
        factory = _make_mock_factory()
        candidates = np.array([[0.0], [1.0], [2.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2])
        assert len(result["rankings"]) == 3
        assert len(result["scores"]) == 3

    def test_lower_sum_ranked_first(self) -> None:
        """Candidate with x=[0] should rank above x=[10] (lower objective)."""
        factory = _make_mock_factory(base_value=0.0, seed_noise=0.001)
        candidates = np.array([[0.0], [10.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2, 3])
        assert result["rankings"][0] == 0  # x=[0.0] is better

    def test_scores_have_multiseed_fields(self) -> None:
        factory = _make_mock_factory()
        candidates = np.array([[0.0], [1.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2])
        score = result["scores"][0]
        assert "per_seed" in score
        assert "mean" in score
        assert "cv" in score
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_multiseed.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement multiseed.py**

```python
# osmose/calibration/multiseed.py
"""Multi-seed validation and candidate re-ranking for OSMOSE calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def validate_multiseed(
    make_objective: Callable[[int], Callable[[np.ndarray], float]],
    x: np.ndarray,
    seeds: Sequence[int] = (42, 123, 7, 999, 2024),
) -> dict:
    """Re-evaluate a candidate across multiple random seeds.

    Args:
        make_objective: Factory(seed) -> objective(x) -> float.
        x: Parameter vector to evaluate.
        seeds: Random seeds to test against.

    Returns:
        Dict with per_seed, mean, std, cv, worst_seed, worst_value.
    """
    per_seed: list[float] = []
    for seed in seeds:
        obj_fn = make_objective(seed)
        per_seed.append(float(obj_fn(x)))

    mean = float(np.mean(per_seed))
    std = float(np.std(per_seed))
    cv = std / mean if mean != 0 else float("inf")
    worst_idx = int(np.argmax(per_seed))

    return {
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "cv": cv,
        "worst_seed": seeds[worst_idx],
        "worst_value": per_seed[worst_idx],
    }


def rank_candidates_multiseed(
    make_objective: Callable[[int], Callable[[np.ndarray], float]],
    candidates: np.ndarray,
    seeds: Sequence[int] = (42, 123, 7, 999, 2024),
) -> dict:
    """Re-rank candidates by mean objective across multiple seeds.

    Args:
        candidates: Array of shape (n_candidates, n_params).

    Returns:
        Dict with rankings (sorted candidate indices) and scores (per-candidate dicts).
    """
    scores: list[dict] = []
    for i in range(len(candidates)):
        score = validate_multiseed(make_objective, candidates[i], seeds)
        scores.append(score)

    means = [s["mean"] for s in scores]
    rankings = list(int(i) for i in np.argsort(means))

    return {"rankings": rankings, "scores": scores}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_multiseed.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```
git add osmose/calibration/multiseed.py tests/test_calibration_multiseed.py
git commit -m "feat(calibration): add multi-seed validation and candidate ranking"
```

---

### Task 4: `surrogate.py` — Cross-validation and fit_score_

**Files:**
- Modify: `osmose/calibration/surrogate.py`
- Create: `tests/test_calibration_surrogate.py`

- [ ] **Step 1: Write tests for cross_validate() and fit_score_**

```python
# tests/test_calibration_surrogate.py
"""Tests for SurrogateCalibrator cross-validation and fit_score_."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.surrogate import SurrogateCalibrator


@pytest.fixture()
def sample_data():
    """Simple 1D function: y = x^2 with noise."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(50, 1))
    y = X[:, 0] ** 2 + rng.normal(0, 1, size=50)
    return X, y


class TestFitScore:
    def test_fit_score_none_before_fit(self) -> None:
        cal = SurrogateCalibrator(param_bounds=[(0, 10)])
        assert cal.fit_score_ is None

    def test_fit_score_set_after_fit(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        cal.fit(X, y)
        assert cal.fit_score_ is not None
        assert isinstance(cal.fit_score_, float)

    def test_fit_score_near_one_for_gp(self, sample_data) -> None:
        """GP is an exact interpolator; in-sample R² should be very high."""
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        cal.fit(X, y)
        assert cal.fit_score_ > 0.99


class TestCrossValidate:
    def test_returns_expected_keys(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=3)
        assert "fold_rmse" in result
        assert "fold_r2" in result
        assert "mean_rmse" in result
        assert "mean_r2" in result
        assert "std_rmse" in result
        assert "std_r2" in result

    def test_fold_count_matches_k(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=5)
        assert len(result["fold_rmse"]) == 5
        assert len(result["fold_r2"]) == 5

    def test_mean_is_average_of_folds(self, sample_data) -> None:
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=3)
        assert result["mean_rmse"] == pytest.approx(np.mean(result["fold_rmse"]))
        assert result["mean_r2"] == pytest.approx(np.mean(result["fold_r2"]))

    def test_reasonable_r2_on_smooth_function(self, sample_data) -> None:
        """y = x^2 + noise is smooth — GP should generalize well."""
        X, y = sample_data
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        result = cal.cross_validate(X, y, k_folds=5)
        assert result["mean_r2"] > 0.8

    def test_raises_if_too_few_samples(self) -> None:
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 4.0])
        cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_restarts_optimizer=0)
        with pytest.raises(ValueError, match="k_folds"):
            cal.cross_validate(X, y, k_folds=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_surrogate.py -v`
Expected: FAIL — `AttributeError: 'SurrogateCalibrator' object has no attribute 'fit_score_'`

- [ ] **Step 3: Modify surrogate.py — add fit_score_ and cross_validate()**

Add `fit_score_` initialization in `__init__`, compute it at the end of `fit()`, and add the `cross_validate()` method. The changes to `osmose/calibration/surrogate.py`:

In `__init__`, after `self._n_restarts_optimizer = n_restarts_optimizer`, add:
```python
        self.fit_score_: float | None = None
```

At the end of `fit()`, after `self._is_fitted = True`, add:
```python
        # Compute in-sample R² as a sanity check
        # Note: y has already been reshaped to 2D earlier in fit(), so y[:, 0] is safe
        means, _ = self.predict(X)
        y_col = y[:, 0]
        ss_res = float(np.sum((y_col - means[:, 0]) ** 2))
        ss_tot = float(np.sum((y_col - np.mean(y_col)) ** 2))
        self.fit_score_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
```

Add new method after `find_optimum()`:
```python
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_folds: int = 5,
        seed: int = 42,
    ) -> dict:
        """K-fold cross-validation of the surrogate.

        Raises ValueError if len(X) < k_folds.
        """
        if len(X) < k_folds:
            raise ValueError(
                f"Need at least k_folds={k_folds} samples, got {len(X)}"
            )

        from sklearn.model_selection import KFold  # type: ignore[import-untyped]

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_rmse: list[float] = []
        fold_r2: list[float] = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_cal = SurrogateCalibrator(
                param_bounds=self.param_bounds,
                n_objectives=y.shape[1],
                n_restarts_optimizer=self._n_restarts_optimizer,
            )
            fold_cal.fit(X_train, y_train)
            means, _ = fold_cal.predict(X_test)

            y_col = y_test[:, 0]
            pred_col = means[:, 0]
            rmse = float(np.sqrt(np.mean((y_col - pred_col) ** 2)))
            ss_res = float(np.sum((y_col - pred_col) ** 2))
            ss_tot = float(np.sum((y_col - np.mean(y_col)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fold_rmse.append(rmse)
            fold_r2.append(r2)

        return {
            "fold_rmse": fold_rmse,
            "fold_r2": fold_r2,
            "mean_rmse": float(np.mean(fold_rmse)),
            "mean_r2": float(np.mean(fold_r2)),
            "std_rmse": float(np.std(fold_rmse)),
            "std_r2": float(np.std(fold_r2)),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_surrogate.py -v`
Expected: all PASS

- [ ] **Step 5: Run existing surrogate-related tests to check no regression**

Run: `.venv/bin/python -m pytest tests/ -k "surrogate or calibration" -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```
git add osmose/calibration/surrogate.py tests/test_calibration_surrogate.py
git commit -m "feat(calibration): add surrogate cross-validation and fit_score_"
```

---

### Task 5: `sensitivity.py` — Multi-objective support

**Files:**
- Modify: `osmose/calibration/sensitivity.py`
- Create: `tests/test_calibration_sensitivity.py`

- [ ] **Step 1: Write tests for 1D backward-compat and 2D multi-objective**

```python
# tests/test_calibration_sensitivity.py
"""Tests for multi-objective Sobol sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.sensitivity import SensitivityAnalyzer


@pytest.fixture()
def analyzer():
    return SensitivityAnalyzer(
        param_names=["a", "b"],
        param_bounds=[(0, 1), (0, 1)],
    )


class TestAnalyze1D:
    """Backward-compatible 1D path."""

    def test_returns_expected_keys(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.sum(samples, axis=1)  # Simple function: y = a + b
        result = analyzer.analyze(Y)
        assert "S1" in result
        assert "ST" in result
        assert "S1_conf" in result
        assert "ST_conf" in result
        assert "param_names" in result
        assert "objective_names" not in result  # 1D path — no objective_names key

    def test_s1_shape(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.sum(samples, axis=1)
        result = analyzer.analyze(Y)
        assert result["S1"].shape == (2,)  # 2 params
        assert result["ST"].shape == (2,)


class TestAnalyze2D:
    """New multi-objective 2D path."""

    def test_returns_objective_names(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([
            np.sum(samples, axis=1),
            np.prod(samples, axis=1),
        ])
        result = analyzer.analyze(Y, objective_names=["sum", "product"])
        assert result["objective_names"] == ["sum", "product"]
        assert result["n_objectives"] == 2

    def test_default_objective_names(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([samples[:, 0], samples[:, 1]])
        result = analyzer.analyze(Y)
        assert result["objective_names"] == ["obj_0", "obj_1"]

    def test_s1_shape_2d(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([
            np.sum(samples, axis=1),
            samples[:, 0] ** 2,
        ])
        result = analyzer.analyze(Y)
        assert result["S1"].shape == (2, 2)  # (n_obj, n_params)
        assert result["ST"].shape == (2, 2)

    def test_n_objectives_key(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([samples[:, 0], samples[:, 1], samples[:, 0] + samples[:, 1]])
        result = analyzer.analyze(Y)
        assert result["n_objectives"] == 3
        assert result["S1"].shape == (3, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_sensitivity.py -v`
Expected: FAIL — 2D tests fail (1D tests may pass since the existing `analyze()` works for 1D)

- [ ] **Step 3: Modify sensitivity.py — extend analyze() for 2D Y**

Replace the `analyze` method in `osmose/calibration/sensitivity.py`:

```python
    def analyze(self, Y: np.ndarray, objective_names: list[str] | None = None) -> dict:
        """Compute Sobol sensitivity indices for one or more objectives.

        Args:
            Y: 1D array (single objective) or 2D (n_samples, n_objectives).
            objective_names: Labels per objective. Defaults to ["obj_0", ...].

        Returns:
            1D: dict with S1, ST, S1_conf, ST_conf, param_names.
            2D: same keys with arrays of shape (n_obj, n_params), plus
                objective_names and n_objectives.
        """
        if Y.ndim == 1:
            result = sobol_analyze.analyze(self.problem, Y)
            return {
                "S1": result["S1"],
                "ST": result["ST"],
                "S1_conf": result["S1_conf"],
                "ST_conf": result["ST_conf"],
                "param_names": self.problem["names"],
            }

        n_obj = Y.shape[1]
        if objective_names is None:
            objective_names = [f"obj_{i}" for i in range(n_obj)]

        all_s1, all_st, all_s1_conf, all_st_conf = [], [], [], []
        for col in range(n_obj):
            result = sobol_analyze.analyze(self.problem, Y[:, col])
            all_s1.append(result["S1"])
            all_st.append(result["ST"])
            all_s1_conf.append(result["S1_conf"])
            all_st_conf.append(result["ST_conf"])

        return {
            "S1": np.array(all_s1),
            "ST": np.array(all_st),
            "S1_conf": np.array(all_s1_conf),
            "ST_conf": np.array(all_st_conf),
            "param_names": self.problem["names"],
            "objective_names": objective_names,
            "n_objectives": n_obj,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_sensitivity.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```
git add osmose/calibration/sensitivity.py tests/test_calibration_sensitivity.py
git commit -m "feat(calibration): extend sensitivity analysis for multi-objective"
```

---

### Task 6: `problem.py` — Evaluation cache + schema validation

**Files:**
- Modify: `osmose/calibration/problem.py`
- Modify: `tests/test_calibration_problem.py`

- [ ] **Step 1: Write tests for cache and schema validation**

Add to `tests/test_calibration_problem.py`:

```python
# --- Cache tests ---

import json
import hashlib

@patch("subprocess.run")
@patch("osmose.results.OsmoseResults")
def test_cache_miss_then_hit(mock_results_cls, mock_subprocess, tmp_path):
    """Second call with same overrides returns cached result without subprocess."""
    mock_subprocess.return_value = MagicMock(returncode=0)
    mock_results_cls.return_value = MagicMock()

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.5],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
        enable_cache=True,
    )
    # Create fake JAR so st_mtime works
    (tmp_path / "fake.jar").write_bytes(b"fake")

    result1 = problem._run_single({"species.k.sp0": "0.3"}, run_id=0)
    assert result1 == [0.5]
    assert mock_subprocess.call_count == 1

    result2 = problem._run_single({"species.k.sp0": "0.3"}, run_id=1)
    assert result2 == [0.5]
    assert mock_subprocess.call_count == 1  # Not called again — cache hit

    stats = problem.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@patch("subprocess.run")
@patch("osmose.results.OsmoseResults")
def test_cache_invalidated_by_jar_change(mock_results_cls, mock_subprocess, tmp_path):
    """Changing JAR file invalidates cache."""
    mock_subprocess.return_value = MagicMock(returncode=0)
    mock_results_cls.return_value = MagicMock()

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    jar = tmp_path / "fake.jar"
    jar.write_bytes(b"v1")

    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.5],
        base_config_path=tmp_path / "config.csv",
        jar_path=jar,
        work_dir=tmp_path,
        enable_cache=True,
    )

    problem._run_single({"species.k.sp0": "0.3"}, run_id=0)
    assert mock_subprocess.call_count == 1

    # Simulate JAR recompilation
    import time
    time.sleep(0.01)
    jar.write_bytes(b"v2")

    problem._run_single({"species.k.sp0": "0.3"}, run_id=1)
    assert mock_subprocess.call_count == 2  # Cache miss — JAR changed


def test_cache_disabled_by_default(tmp_path):
    """enable_cache defaults to False."""
    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    assert problem._enable_cache is False


def test_clear_cache(tmp_path):
    """clear_cache removes the cache directory contents."""
    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
        enable_cache=True,
    )
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    (cache_dir / "test.json").write_text("{}")
    problem.clear_cache()
    assert not list(cache_dir.iterdir())


# --- Schema validation tests ---


def test_validate_overrides_catches_bad_value(tmp_path):
    """Schema validation rejects values outside [min_val, max_val]."""
    from osmose.schema.registry import ParameterRegistry
    from osmose.schema.base import OsmoseField, ParamType

    registry = ParameterRegistry()
    registry.register(OsmoseField(
        key_pattern="species.k.sp{idx}",
        param_type=ParamType.FLOAT,
        min_val=0.01,
        max_val=1.0,
        indexed=True,
        category="species",
        description="Growth rate",
    ))

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.01, upper_bound=1.0)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
        registry=registry,
    )

    with pytest.raises(ValueError, match="species.k.sp0"):
        problem._validate_overrides({"species.k.sp0": "999.0"})


def test_validate_overrides_skipped_when_no_registry(tmp_path):
    """No registry means no validation — backward-compatible."""
    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    # Should not raise
    problem._validate_overrides({"species.k.sp0": "999.0"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -k "cache or validate_overrides" -v`
Expected: FAIL — new parameters/methods don't exist yet

- [ ] **Step 3: Modify problem.py — add cache and validation**

Changes to `osmose/calibration/problem.py`:

Add imports at top:
```python
import hashlib
import json
import os
import tempfile
```

Modify `__init__` — add new params after `n_parallel`:
```python
        enable_cache: bool = False,
        cache_dir: Path | None = None,
        registry: "ParameterRegistry | None" = None,
```

Add in `__init__` body after `self.n_parallel = max(1, n_parallel)`:
```python
        self._enable_cache = enable_cache
        self._cache_dir = cache_dir or (self.work_dir / ".cache")
        self._registry = registry
        self._cache_hits = 0
        self._cache_misses = 0
        # Pre-compute base config hash for cache keys
        self._base_config_hash = ""
        if enable_cache and base_config_path.exists():
            self._base_config_hash = hashlib.sha256(
                base_config_path.read_bytes()
            ).hexdigest()[:16]
```

Add new methods after `cleanup_run`:
```python
    def _cache_key(self, overrides: dict[str, str]) -> str:
        """Deterministic hash of overrides + JAR mtime + base config hash."""
        parts = sorted(overrides.items())
        try:
            jar_mtime = str(self.jar_path.stat().st_mtime)
        except OSError:
            jar_mtime = "missing"
        raw = f"{parts}|{jar_mtime}|{self._base_config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def cache_stats(self) -> dict:
        """Returns cache hit/miss counts and size."""
        size_mb = 0.0
        if self._cache_dir.is_dir():
            size_mb = sum(f.stat().st_size for f in self._cache_dir.iterdir()) / (1024 * 1024)
        return {"hits": self._cache_hits, "misses": self._cache_misses, "size_mb": size_mb}

    def clear_cache(self) -> None:
        """Remove all cached evaluations."""
        if self._cache_dir.is_dir():
            for f in self._cache_dir.iterdir():
                f.unlink(missing_ok=True)

    def _validate_overrides(self, overrides: dict[str, str]) -> None:
        """Validate overrides against the schema registry.

        Note: overrides values are strings (from OSMOSE config format).
        We must coerce numeric values to float/int before passing to the
        registry, since validate_value() compares with ``<`` / ``>``.
        """
        if self._registry is None:
            return
        from osmose.schema.base import ParamType

        coerced: dict[str, object] = {}
        for k, v in overrides.items():
            field = self._registry.match_field(k)
            if field and field.param_type == ParamType.FLOAT:
                coerced[k] = float(v)
            elif field and field.param_type == ParamType.INT:
                coerced[k] = int(v)
            else:
                coerced[k] = v
        errors = self._registry.validate(coerced)
        if errors:
            raise ValueError(
                f"Override validation failed ({len(errors)} errors):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
```

Modify `_run_single` — add cache check at start (after key/value validation, before run_dir creation):
```python
        # Schema validation
        self._validate_overrides(overrides)

        # Cache check
        if self._enable_cache:
            key = self._cache_key(overrides)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                self._cache_hits += 1
                cached = json.loads(cache_file.read_text())
                return cached["objectives"]
```

After computing objectives (before `return obj_values`), add cache write:
```python
        # Cache write (atomic rename)
        if self._enable_cache:
            key = self._cache_key(overrides)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{key}.json"
            fd, tmp_file = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"objectives": obj_values}, f)
                os.replace(tmp_file, str(cache_file))
            except OSError:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
            self._cache_misses += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -v`
Expected: all PASS (new + existing)

- [ ] **Step 5: Commit**

```
git add osmose/calibration/problem.py tests/test_calibration_problem.py
git commit -m "feat(calibration): add evaluation cache and schema validation to problem"
```

---

### Task 7: `__init__.py` — Export new public API

**Files:**
- Modify: `osmose/calibration/__init__.py`

- [ ] **Step 1: Update __init__.py with new exports**

Replace the contents of `osmose/calibration/__init__.py`:

```python
"""OSMOSE calibration module — optimization, surrogate modelling, and sensitivity analysis."""

from osmose.calibration.objectives import (
    biomass_rmse,
    abundance_rmse,
    diet_distance,
    normalized_rmse,
    yield_rmse,
    catch_at_size_distance,
    size_at_age_rmse,
    weighted_multi_objective,
)
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem, Transform
from osmose.calibration.surrogate import SurrogateCalibrator
from osmose.calibration.sensitivity import SensitivityAnalyzer
from osmose.calibration.targets import BiomassTarget, load_targets
from osmose.calibration.losses import (
    banded_log_ratio_loss,
    stability_penalty,
    worst_species_penalty,
    make_banded_objective,
)
from osmose.calibration.multiseed import validate_multiseed, rank_candidates_multiseed

__all__ = [
    "biomass_rmse",
    "abundance_rmse",
    "diet_distance",
    "normalized_rmse",
    "yield_rmse",
    "catch_at_size_distance",
    "size_at_age_rmse",
    "weighted_multi_objective",
    "FreeParameter",
    "Transform",
    "OsmoseCalibrationProblem",
    "SurrogateCalibrator",
    "SensitivityAnalyzer",
    "BiomassTarget",
    "load_targets",
    "banded_log_ratio_loss",
    "stability_penalty",
    "worst_species_penalty",
    "make_banded_objective",
    "validate_multiseed",
    "rank_candidates_multiseed",
]
```

- [ ] **Step 2: Verify all imports resolve**

Run: `.venv/bin/python -c "from osmose.calibration import BiomassTarget, load_targets, banded_log_ratio_loss, stability_penalty, worst_species_penalty, make_banded_objective, validate_multiseed, rank_candidates_multiseed; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```
git add osmose/calibration/__init__.py
git commit -m "feat(calibration): export new public API from __init__.py"
```

---

### Task 8: Update biomass_targets.csv

**Files:**
- Modify: `data/baltic/reference/biomass_targets.csv`

- [ ] **Step 1: Update the CSV with reference_point_type and metadata**

Replace the contents of `data/baltic/reference/biomass_targets.csv`:

```csv
#! version: 1.0
#! last_updated: 2026-04-15
#! ices_advice_year: 2022-2023
# Baltic Sea equilibrium biomass targets for calibration
# Sources: ICES stock assessments (2018-2022 averages), FishBase, published literature
# Biomass type: total stock biomass (not SSB), whole Baltic model domain (10-30E, 54-66N)
# These are order-of-magnitude targets with acceptable ranges for calibration
# Weight: 1.0=high (well-assessed pelagics), 0.5=medium, 0.2=low (poorly resolved at grid scale)
species,target_tonnes,lower_tonnes,upper_tonnes,weight,reference_point_type,source,notes
cod,120000,60000,250000,1.0,ssb,ICES SD24-32 eastern+western Baltic 2018-2022; eastern SSB ~60-77kt post-collapse,Post-2015 collapse state; total biomass ~1.5-2x SSB
herring,1500000,800000,3000000,1.0,biomass,ICES aggregate Baltic herring complex (SD22-32 incl. Gulf of Bothnia),Central Baltic SSB ~450kt; total across all stocks ~1-2Mt
sprat,1500000,800000,2500000,1.0,biomass,ICES Baltic sprat SD22-32; SSB ~1.0Mt in 2022 (doi:10.17895/ices.advice.19453856),Total biomass > SSB; raised from 800kt to match ICES
flounder,50000,20000,100000,0.5,biomass,ICES Baltic flounder combined,Multiple management units; rough estimate
perch,20000,8000,50000,0.2,biomass,Literature estimate for coastal Baltic,Poorly assessed at basin scale; coarse grid under-resolves habitat
pikeperch,10000,4000,25000,0.2,biomass,Literature estimate for coastal Baltic,Concentrated in estuaries/lagoons; coarse grid under-resolves
whitefish,15000,5000,30000,0.2,biomass,Literature estimate for northern Baltic,Gulf of Bothnia primarily; coarse grid under-resolves
stickleback,200000,50000,500000,0.2,biomass,Olsson et al. 2019 (doi:10.1093/icesjms/fsz078); highly uncertain,Boom-bust dynamics; wide range acceptable
```

- [ ] **Step 2: Verify load_targets reads updated CSV**

Run: `.venv/bin/python -c "from osmose.calibration.targets import load_targets; from pathlib import Path; t, m = load_targets(Path('data/baltic/reference/biomass_targets.csv')); print(f'{len(t)} targets, metadata: {m}'); print(f'cod ref type: {t[0].reference_point_type}')"`
Expected: `8 targets, metadata: {'version': '1.0', 'last_updated': '2026-04-15', 'ices_advice_year': '2022-2023'}` and `cod ref type: ssb`

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_targets.py -v`
Expected: all PASS (including the smoke test that loads the real CSV)

- [ ] **Step 4: Commit**

```
git add data/baltic/reference/biomass_targets.csv
git commit -m "data: add reference_point_type and metadata to Baltic targets CSV"
```

---

### Task 9: Full regression test + lint

**Files:** None — validation only.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass, no regressions

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/calibration/ tests/test_calibration_*.py`
Expected: no errors

- [ ] **Step 3: Run format check**

Run: `.venv/bin/ruff format --check osmose/calibration/ tests/test_calibration_*.py`
Expected: all files formatted correctly (if not, run `.venv/bin/ruff format osmose/calibration/ tests/test_calibration_*.py` and commit)

- [ ] **Step 4: Verify all new exports are importable**

Run: `.venv/bin/python -c "import osmose.calibration; print(sorted(osmose.calibration.__all__))"`
Expected: List of 22 names including all new exports

- [ ] **Step 5: Commit any formatting fixes**

```
git add -u
git commit -m "style: format calibration library gap files"
```
(Skip if no changes needed.)
