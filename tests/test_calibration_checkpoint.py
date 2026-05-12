from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from osmose.calibration.checkpoint import (
    CalibrationCheckpoint,
    MAX_CHECKPOINT_BYTES,
    default_results_dir,
)


def test_default_results_dir_resolves_to_baltic_calibration_results():
    """default_results_dir points at the Baltic results dir, package-root-resolved."""
    p = default_results_dir()
    assert isinstance(p, Path)
    assert p.parts[-3:] == ("data", "baltic", "calibration_results")


def test_max_checkpoint_bytes_is_1mib():
    """1 MiB ceiling for read_checkpoint's size guard."""
    assert MAX_CHECKPOINT_BYTES == 1_048_576


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
