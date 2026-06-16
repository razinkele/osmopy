"""Unit tests for the pure image-diff core. Runs in the normal suite (numpy + Pillow,
no browser). Pillow is in [dev]; the find_spec guard is defensive."""

from __future__ import annotations

import io
from importlib.util import find_spec

import numpy as np
import pytest

if find_spec("PIL") is None:  # pragma: no cover - Pillow is in [dev]
    pytest.skip("Pillow not installed", allow_module_level=True)

from PIL import Image

from tests._visual_compare import compare_images


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8"), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _solid(h: int, w: int, color=(10, 20, 30)) -> np.ndarray:
    a = np.zeros((h, w, 3), dtype="uint8")
    a[:, :] = color
    return a


def test_identical_images_pass():
    png = _png(_solid(20, 20))
    passed, metrics, _ = compare_images(png, png)
    assert passed is True
    assert metrics["diff_ratio"] == 0.0
    assert metrics["diff_pixels"] == 0
    assert metrics["mean_delta"] == 0.0


def test_subthreshold_localized_noise_passes():
    base = _solid(40, 40, (100, 100, 100))
    act = base.copy()
    act[0:2, 0:2] = (102, 100, 100)  # 4 px, per-channel delta 2 < threshold 4
    passed, metrics, _ = compare_images(_png(base), _png(act))
    assert passed is True
    assert metrics["diff_pixels"] == 0


def test_dimension_mismatch_fails():
    passed, metrics, _ = compare_images(_png(_solid(20, 20)), _png(_solid(21, 20)))
    assert passed is False
    assert metrics["reason"] == "dimension"


def test_localized_block_fails_ratio_and_highlights_red():
    base = _solid(20, 20, (0, 0, 0))
    act = base.copy()
    act[0:10, :] = (255, 255, 255)  # 50% of pixels, delta 255
    passed, metrics, diff_png = compare_images(_png(base), _png(act), max_ratio=0.001)
    assert passed is False
    assert "ratio" in metrics["reasons"]
    diff = np.asarray(Image.open(io.BytesIO(diff_png)).convert("RGB"))
    assert (diff[0:10] == (255, 0, 0)).all()


def test_uniform_global_recolor_fails_via_mean():
    # Every pixel shifts +3 on every channel: max-channel delta 3 < threshold 4, so
    # diff_pixels == 0 and the per-pixel/ratio/floor gates all see "no change". Only the
    # mean-delta gate catches this Bootstrap-style uniform recolor. This is the headline
    # efficacy test.
    base = _solid(30, 30, (100, 100, 100))
    act = (base.astype("int16") + 3).clip(0, 255).astype("uint8")
    passed, metrics, _ = compare_images(
        _png(base), _png(act), max_ratio=0.9, max_pixels=10_000, mean_threshold=1.0
    )
    assert passed is False
    assert metrics["diff_pixels"] == 0
    assert metrics["mean_delta"] == pytest.approx(3.0, abs=0.01)
    assert "mean" in metrics["reasons"]


def test_absolute_pixel_floor_fails_under_ratio():
    # A small strong-change cluster: over max_pixels but under max_ratio.
    base = _solid(40, 40, (0, 0, 0))
    act = base.copy()
    act[0:3, 0:3] = (255, 255, 255)  # 9 px of 1600
    passed, metrics, _ = compare_images(
        _png(base), _png(act), max_ratio=0.9, max_pixels=5, mean_threshold=100.0
    )
    assert passed is False
    assert metrics["diff_pixels"] == 9
    assert "pixels" in metrics["reasons"]
    assert "ratio" not in metrics["reasons"]


def test_ratio_boundary():
    base = _solid(10, 10, (0, 0, 0))  # 100 px
    act = base.copy()
    act[0, 0] = (255, 255, 255)  # 1 px = 1%
    # isolate ratio: floor + mean disabled
    assert (
        compare_images(
            _png(base), _png(act), max_ratio=0.02, max_pixels=10_000, mean_threshold=100.0
        )[0]
        is True
    )
    assert (
        compare_images(
            _png(base), _png(act), max_ratio=0.005, max_pixels=10_000, mean_threshold=100.0
        )[0]
        is False
    )
