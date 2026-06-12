"""Skip guards for tests that read gitignored, generated model artifacts.

Some integration tests read real engine/calibration outputs directly:

  * ``data/eec_full/output/**``      — produced by running the EEC model
  * ``data/baltic/calibration_results/phase12_results.json`` — a calibration artifact

Both trees are gitignored (``.gitignore``: ``data/eec_full/output/``,
``data/baltic/calibration_results/``), so they exist in a developer tree that
has run the model but **never** in a clean CI checkout. Tests that read them
must SKIP when the artifact is absent rather than hard-fail — otherwise CI is
permanently red for an environment reason, not a code defect.

This mirrors the existing ``find_spec`` import guards in ``conftest.py`` that
skip the playwright/hypothesis suites when those optional deps are missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEC_OUTPUT = PROJECT_ROOT / "data" / "eec_full" / "output"
BALTIC_PHASE12 = PROJECT_ROOT / "data" / "baltic" / "calibration_results" / "phase12_results.json"


def require_eec_output(pattern: str) -> None:
    """Skip the calling test unless a file matching ``pattern`` exists anywhere
    under the EEC output tree.

    ``pattern`` is a recursive glob fragment such as ``"*dietMatrix*"`` or
    ``"eec_biomassDistribBySize*"``. The match is recursive so files under
    ``Trophic/``, ``Indicators/``, etc. are found regardless of subdirectory.
    """
    if not EEC_OUTPUT.is_dir() or not any(EEC_OUTPUT.rglob(pattern)):
        pytest.skip(
            f"EEC model output missing ('{pattern}' under {EEC_OUTPUT}); "
            "this output is gitignored and absent in clean checkouts — "
            "run the EEC model to generate it."
        )


def require_baltic_phase12() -> None:
    """Skip the calling test unless the gitignored phase-12 calibration results
    exist (the prerequisite for reconstructing phase-13 params)."""
    if not BALTIC_PHASE12.is_file():
        pytest.skip(
            f"{BALTIC_PHASE12} missing; phase-12 calibration results are "
            "gitignored and absent in clean checkouts — run phase-12 "
            "calibration to generate."
        )
