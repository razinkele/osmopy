from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from osmose.calibration.checkpoint import (
    CalibrationCheckpoint,
    CheckpointReadResult,
    MAX_CHECKPOINT_BYTES,
    default_results_dir,
    read_checkpoint,
    write_checkpoint,
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


def test_invariant_param_keys_unique():
    """Duplicate param_keys must be rejected; set-equality with best_parameters
    would otherwise silently accept ('a', 'a') matched to {'a': 1.0}."""
    kwargs = _valid_checkpoint_kwargs()
    kwargs["param_keys"] = ("k_a", "k_a")  # duplicate
    kwargs["best_parameters"] = {"k_a": 0.5}
    kwargs["best_x_log10"] = (-0.3, -0.3)
    kwargs["bounds_log10"] = {"k_a": (-1.0, 0.0)}
    with pytest.raises(ValueError, match="param_keys"):
        CalibrationCheckpoint(**kwargs)


def test_invariant_banded_loss_requires_banded_targets():
    """proxy_source='banded_loss' is meaningless without banded_targets;
    the proxy table renderer (§8) crashes on banded_targets[species]
    if banded_targets is None."""
    kwargs = _valid_checkpoint_kwargs()
    kwargs["banded_targets"] = None
    # per_species_residuals / sim_biomass / species_labels / proxy_source
    # all stay as their valid_kwargs values (banded_loss configured)
    with pytest.raises(ValueError, match="banded_targets"):
        CalibrationCheckpoint(**kwargs)


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
    kwargs["best_fun"] = np.float64(3.14)
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
    path.write_bytes(b"{}" + b" " * (MAX_CHECKPOINT_BYTES))
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
    probe_writable(tmp_path)


def test_probe_writable_does_not_leak_sentinel(tmp_path):
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


from osmose.calibration.checkpoint import liveness_state


def test_liveness_state_live_boundary():
    assert liveness_state(age_seconds=30.0) == "live"
    assert liveness_state(age_seconds=0.0) == "live"


def test_liveness_state_stalled_boundary_60s():
    assert liveness_state(age_seconds=61.0) == "stalled"
    assert liveness_state(age_seconds=120.0) == "stalled"


def test_liveness_state_stalled_to_idle_boundary_300s():
    assert liveness_state(age_seconds=299.0) == "stalled"
    assert liveness_state(age_seconds=301.0) == "idle"
    assert liveness_state(age_seconds=3600.0) == "idle"


def test_liveness_state_rejects_negative_age():
    assert liveness_state(age_seconds=-10.0) == "idle"


from osmose.calibration.checkpoint import LiveSnapshot


def test_live_snapshot_construction():
    sentinel = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    snap = LiveSnapshot(active=sentinel, other_live_paths=(), snapshot_monotonic=42.0)
    assert snap.active is sentinel
    assert snap.other_live_paths == ()


def test_live_snapshot_replace_via_dataclasses_replace():
    """Frozen dataclass: dataclasses.replace produces a new instance."""
    sentinel = CheckpointReadResult(kind="no_run", checkpoint=None, error_summary=None)
    snap = LiveSnapshot(active=sentinel, other_live_paths=(), snapshot_monotonic=42.0)
    snap2 = dataclasses.replace(snap, snapshot_monotonic=99.0)
    assert snap2.snapshot_monotonic == 99.0
    assert snap2.active is sentinel
    assert snap.snapshot_monotonic == 42.0
