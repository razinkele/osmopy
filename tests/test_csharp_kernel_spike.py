"""Kernel spike, plan task 0.6, checks 1-4: the C# mortality kernel against the Numba oracle.

A fixture is captured from a short EEC run (``scripts/csharp/make_fixtures.py``), twice:
- with the step's real rates;
- with ``--zero-rates``, which leaves no ``exp`` anywhere in the loop (tier E).

The probe subprocess then runs every C# layout x prefilter on the oracle's injected draws.
- **Check 1** records whether ``Math.Exp`` equals Numba's ``exp`` bit for bit.
- **Check 2:** with zero rates, every output equals the oracle exactly.
- **Check 3:** with real rates, exactly if check 1 found the two ``exp`` identical, else within
  tier L (1e-9 relative).
- **Check 4:** all layouts and prefilters give bit-identical results.

Needs the built engine; skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osmose.engine_cs import csharp_engine_available

ROOT = Path(__file__).resolve().parents[1]
REQUIRE = os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1"
DEMO, STEP = "eec_full", 5


def _need_engine() -> None:
    ok, why = csharp_engine_available()
    if not ok:
        if REQUIRE:
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(
            f"C# engine unavailable ({why}). To run: bash scripts/build_csharp_engine.sh, "
            "then OSMOSE_REQUIRE_CSHARP=1 pytest tests/test_csharp_kernel_spike.py"
        )


def _run(args: list[str], timeout: int) -> str:
    env = {**os.environ, "NUMBA_NUM_THREADS": "1"}
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)
    assert proc.returncode == 0, f"{args[1]} failed:\n{proc.stderr[-4000:]}"
    return proc.stdout


@pytest.fixture(scope="module")
def probes(tmp_path_factory) -> dict:
    _need_engine()
    d = tmp_path_factory.mktemp("kernel")
    out = {}
    for zero in (True, False):
        cmd = [
            sys.executable,
            "scripts/csharp/make_fixtures.py",
            "--demo",
            DEMO,
            "--step",
            str(STEP),
            "--out",
            str(d),
        ]
        _run(cmd + (["--zero-rates"] if zero else []), timeout=600)
        fixture = d / f"{DEMO}_step{STEP}{'_zero' if zero else ''}.bin"
        line = _run([sys.executable, "tests/_csharp_kernel_probe.py", str(fixture)], timeout=600)
        out["zero" if zero else "real"] = json.loads(line.strip().splitlines()[-1])
    return out


def _assert_exact(runs: dict) -> None:
    for variant, rows in runs.items():
        bad = {k: v for k, v in rows.items() if v["n_diff"]}
        assert not bad, f"{variant} differs from the oracle: {bad}"


def test_check2_zero_rates_equal_the_oracle_bit_for_bit(probes):
    _assert_exact(probes["zero"]["runs"])


def test_check3_real_rates(probes):
    n_args, n_diff, max_ulp = (int(x) for x in probes["real"]["libm"])
    assert n_args > 1_000_000
    if n_diff == 0:
        _assert_exact(probes["real"]["runs"])
    else:  # tier L, with the worst error recorded in the assertion message
        worst = max(r["max_rel"] for rows in probes["real"]["runs"].values() for r in rows.values())
        assert worst <= 1e-9, (
            f"exp differs on {n_diff}/{n_args} args (max {max_ulp} ulp); worst rel {worst}"
        )


@pytest.mark.parametrize("kind", ["zero", "real"])
def test_check4_layouts_and_prefilters_are_bit_identical(probes, kind):
    digests = probes[kind]["digests"]
    assert len(digests) == 6
    assert len(set(digests.values())) == 1, digests
