"""Report Python-engine divergence: 365-step (subsample) vs 24-step (bin-average) BoB.
This is the DELIBERATE forcing change (A4.3) -- characterized, not gated. Run after Task 4."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scripts.native_440_parity import run_outputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    old = run_outputs(ROOT / "data" / "examples_433_orig", years=5, seed=42)  # 365-step
    new = run_outputs(ROOT / "data" / "examples", years=5, seed=42)  # 24-step
    print(f"{'metric':<12}{'max|rel|':>12}{'median|rel|':>14}")
    for k in sorted(set(old) & set(new)):
        a, b = old[k].ravel(), new[k].ravel()
        n = min(a.size, b.size)
        rel = np.abs(a[:n] - b[:n]) / np.maximum(np.abs(a[:n]), 1e-30)
        print(f"{k:<12}{np.nanmax(rel):>12.3f}{np.nanmedian(rel):>14.3f}")


if __name__ == "__main__":
    main()
