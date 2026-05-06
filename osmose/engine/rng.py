"""Per-species deterministic RNG factory.

Reproducibility scope (Phase 3 — RNG documentation):

`fixed=True` gives **Python-side reproducible** outputs only. Running the
same config twice with the same seed under the Python engine produces
bit-equal results. It does NOT produce results bit-equal to the Java
engine. NumPy's `default_rng` uses PCG64; the Java engine uses MT19937
(legacy ``java.util.Random``). The streams diverge at the first draw.

Cross-engine numerical equivalence is documented as "within 1 OoM" by
the parity test suite (14/14 EEC, 8/8 Bay of Biscay), but byte-equivalent
outputs across engines are impossible without reimplementing the MT19937
stream in NumPy — out of scope for this port. If bit-exact reproducibility
against Java is required, set `OsmoseCalibrationProblem(use_java_engine=True)`
and run the Java subprocess path.
"""

from __future__ import annotations

import numpy as np


def build_rng(seed: int, n_species: int, fixed: bool) -> list[np.random.Generator]:
    """Build per-species RNG generators.

    When fixed=False: all species share a single Generator
        (non-deterministic per-species ordering).
    When fixed=True: each species gets a reproducible independent Generator
        via SeedSequence (species ordering has no effect on results).

    See module docstring for the cross-engine caveat (PCG64 != MT19937).
    """
    if not fixed:
        shared = np.random.default_rng(seed)
        return [shared] * n_species
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_species)
    return [np.random.default_rng(child) for child in children]
