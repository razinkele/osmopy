"""`persists` must describe the EQUILIBRIUM, not the seeding transient.

The criterion tested the minimum over the WHOLE run, which is dominated by the seeding bootstrap. The
2026-08-01 seeding A/B showed the consequence: two arms with final-decade means within +-5% of each
other scored 2/9 vs 6/9, purely because one seeded more eggs and so dipped less deeply during
initialisation. cod_east dipping to 17 t before settling at ~83 kt INSIDE its envelope was being
reported as a collapse.

Scoping the minimum to the final decade makes `persists` an equilibrium statement, consistent with
`in_envelope`, which already uses the final-decade mean.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    d = PROJECT_ROOT / "scripts"
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(
        "baltic_stability_certify", d / "baltic_stability_certify.py"
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["baltic_stability_certify"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cert():
    return _load()


def test_deep_seeding_transient_does_not_read_as_collapse(cert):
    """A stock that dips near zero during bootstrap but recovers into envelope must PASS."""
    lo, hi = cert.ENVELOPE["cod_east"]
    # 10 bootstrap steps crashing to 17 t, then 15 steps settled at ~83 kt (inside 60k-85k)
    series = [50000.0, 5000.0, 500.0, 17.0, 200.0, 3000.0, 20000.0, 50000.0, 70000.0, 80000.0]
    series += [83000.0] * 15
    row = cert._species_row(pd.DataFrame({"cod_east": series}), "cod_east")

    assert row["in_envelope"], "final-decade mean is inside envelope"
    assert row["persists"], (
        f"a recovered stock must not read as collapsed; min used was {row['min']}"
    )


def test_genuine_late_collapse_still_fails(cert):
    """A stock that is healthy early and dies late must still FAIL — the fix must not mask real collapse."""
    series = [83000.0] * 15 + [40000.0, 10000.0, 2000.0, 300.0, 40.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    row = cert._species_row(pd.DataFrame({"cod_east": series}), "cod_east")
    assert not row["persists"], "a late collapse must still be detected"


def test_reported_min_is_the_final_decade_min(cert):
    series = [1.0] * 20 + [50000.0] * 10
    row = cert._species_row(pd.DataFrame({"cod_east": series}), "cod_east")
    assert row["min"] == pytest.approx(50000.0), (
        "min must describe the final decade, not the bootstrap"
    )
