"""Fixed-seed parity for every per-step output the core parity baseline omits.

``tests/test_engine_parity.py`` compares biomass, abundance and mortality by cause. Output
COLLECTORS never feed back into the simulation, so a change to one leaves all three untouched:
on 2026-11-14 doubling both outputs of ``_collect_by_life_stage`` left that suite green (perf plan
item 4). This module closes the gap. It compares the by-stage counts and rates, the age and size
distributions, diet by species, mean size, mean trophic level and yield against a committed
baseline, at the same fixed seed and under the same local-only exact-match policy.

Regenerate after an INTENTIONAL change to an output:
    .venv/bin/python scripts/save_parity_baseline.py --outputs --config bob
    .venv/bin/python scripts/save_parity_baseline.py --outputs --config eec

Outputs that neither config enables (SSB, carbon flux, bioenergetics) are not covered here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.save_parity_baseline import flatten_outputs, run_outputs
from tests.test_engine_parity import DEFAULT_SEED, DEFAULT_YEARS, _exact_match_local_only

BASELINE_DIR = Path(__file__).parent / "baselines"


def _baseline_path(config: str) -> Path:
    return BASELINE_DIR / f"parity_outputs_{config}_{DEFAULT_YEARS}yr_seed{DEFAULT_SEED}.npz"


@pytest.fixture(scope="module", params=["bob", "eec"])
def baseline_and_current(request):
    path = _baseline_path(request.param)
    if not path.exists():
        pytest.skip(f"outputs baseline not found: {path.name}")
    with np.load(path) as data:
        baseline = {k: data[k] for k in data.files}
    outputs, _ = run_outputs(DEFAULT_YEARS, DEFAULT_SEED, request.param)
    return request.param, baseline, flatten_outputs(outputs)


@_exact_match_local_only
def test_same_output_fields(baseline_and_current):
    config, baseline, current = baseline_and_current
    assert sorted(current) == sorted(baseline), f"{config}: output fields changed"


@_exact_match_local_only
def test_every_output_matches_the_baseline(baseline_and_current):
    config, baseline, current = baseline_and_current
    mismatched = []
    for key in sorted(set(baseline) & set(current)):
        if baseline[key].shape != current[key].shape or not np.array_equal(
            baseline[key], current[key], equal_nan=True
        ):
            mismatched.append(key)
    assert not mismatched, f"{config}: outputs differ from the baseline: {mismatched}"


def test_the_baseline_covers_the_life_stage_outputs():
    """Guard on the fixture: the gap this module closes must stay closed."""
    for config in ("bob", "eec"):
        path = _baseline_path(config)
        if not path.exists():
            pytest.skip(f"outputs baseline not found: {path.name}")
        with np.load(path) as data:
            files = set(data.files)
        assert {"abundance_by_stage", "mortality_by_cause_stage"} <= files, config
