"""Pure image-comparison core for visual-regression tests.

numpy + Pillow only -- NO playwright import, so the unit tests can run in the normal CI
suite. The browser harness (tests/_visual_support.py) imports this.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _to_rgb_array(png_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.int16)


def compare_images(
    baseline_png: bytes,
    actual_png: bytes,
    *,
    threshold: int = 4,
    # The three tolerances below are conservative committed defaults; the authoritative
    # gate values are supplied by assert_clip_snapshot and tuned (pre-merge) against the
    # first CONTAINER baselines. compare_images is called directly only by its unit tests,
    # which pass tolerances explicitly to isolate each gate.
    max_ratio: float = 0.002,
    max_pixels: int = 800,
    mean_threshold: float = 1.0,
) -> tuple[bool, dict, bytes]:
    """Compare two PNG byte strings on three OR-ed failure conditions.

    A pixel "differs" when its max per-channel absolute difference exceeds ``threshold``
    (absorbs sub-pixel antialiasing). Fails if ANY of:
      - ratio:  differing/total > max_ratio           (gross localized change)
      - pixels: differing > max_pixels                (small glyph-level shift)
      - mean:   mean(|baseline-actual|) > mean_threshold
                (uniform sub-threshold global recolor -- a Bootstrap/theme bump moves
                 every pixel a little and trips ZERO per-pixel counts; only this catches it)

    Returns ``(passed, metrics, diff_png)``. On a dimension mismatch returns
    ``(False, {"reason": "dimension", ...}, actual_png)``. Otherwise ``metrics`` carries
    ``diff_ratio``, ``diff_pixels``, ``mean_delta``, and ``reasons`` (list of fired gates);
    ``diff_png`` highlights differing pixels in red (#FF0000) over a dimmed baseline.
    """
    base = _to_rgb_array(baseline_png)
    act = _to_rgb_array(actual_png)
    if base.shape != act.shape:
        return (
            False,
            {"reason": "dimension", "baseline_shape": base.shape, "actual_shape": act.shape},
            actual_png,
        )

    abs_delta = np.abs(base - act)
    per_pixel = abs_delta.max(axis=2)
    differing = per_pixel > threshold
    diff_pixels = int(differing.sum())
    total = int(differing.size)
    diff_ratio = diff_pixels / total
    mean_delta = float(abs_delta.mean())

    reasons: list[str] = []
    if diff_ratio > max_ratio:
        reasons.append("ratio")
    if diff_pixels > max_pixels:
        reasons.append("pixels")
    if mean_delta > mean_threshold:
        reasons.append("mean")

    highlight = (base // 3).astype(np.uint8)
    highlight[differing] = (255, 0, 0)
    buf = io.BytesIO()
    Image.fromarray(highlight, mode="RGB").save(buf, format="PNG")

    metrics = {
        "diff_ratio": diff_ratio,
        "diff_pixels": diff_pixels,
        "mean_delta": mean_delta,
        "reasons": reasons,
    }
    return (not reasons, metrics, buf.getvalue())
