"""Bridge spike, plan task 0.2: can pythonnet carry the C# engine's contract? (gate G0)

Checks 1-6 run in ONE subprocess (``tests/_csharp_bridge_probe.py``) so the CLR never enters
the pytest process, which may later fork. They need the built engine and skip without it,
unless ``OSMOSE_REQUIRE_CSHARP=1`` (set by the CI ``csharp-engine`` job), which turns the skip
into a failure. A skip is a green, and the one job meant to run these must not skip them.
Check 7 is pure Python and always runs.
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


@pytest.fixture(scope="module")
def probe() -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if REQUIRE:
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(
            f"C# engine unavailable ({why}). To run: bash scripts/build_csharp_engine.sh, "
            "then OSMOSE_REQUIRE_CSHARP=1 pytest tests/test_csharp_bridge_spike.py"
        )
    env = {k: v for k, v in os.environ.items() if k != "DOTNET_ROOT"}  # proves 9.4's fallback
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "_csharp_bridge_probe.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-4000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_probe_measured_this_checkout(probe):
    assert Path(probe["osmose_file"]).resolve().is_relative_to(ROOT)


def test_1_hosts_net10_without_dotnet_root_in_the_environment(probe):
    assert probe["load"]["framework"].startswith(".NET 10."), probe["load"]
    assert probe["load"]["dotnet_root"] not in ("", "None")


def test_2_numpy_buffer_modified_in_place(probe):
    assert probe["in_place"] == {"values": [2.0, 4.0, 6.0, 8.0, 10.0], "same_buffer": True}


def test_3_gil_released_during_a_call(probe):
    # 2 s call, 1 ms ticks: ~2000 if the GIL is released, 0 if it is held.
    assert probe["gil"]["ticks_during_call"] >= 100, probe["gil"]


def test_4_control_block_cancel_stops_within_one_iteration(probe):
    c = probe["cancel"]
    assert c["joined"], c
    assert c["returned"] >= 50, c
    assert c["returned"] <= c["seen_after_cancel"] + 1, c


def test_5_dotnet_exception_reaches_python_with_its_message(probe):
    e = probe["exception"]
    assert e["raised"], e
    assert "spike-message-7f3a" in e["text"], e
    assert "InvalidOperationException" in e["type"] + e["text"], e


def test_6_two_threads_call_at_once(probe):
    c = probe["concurrent"]
    assert c["sums"] == c["expected"], c
    # Each call sleeps 500 ms inside C#: overlapped ~0.5 s, serialized at least 1.0 s (the
    # Py.GIL mutation measured 1.002 s). 0.9 leaves a margin on both sides.
    assert c["wall_s"] < 0.9, c


def test_7_importing_engine_cs_does_not_load_the_clr():
    code = (
        "import sys, osmose.engine_cs as m; "
        "print(m.__file__, 'clr' in sys.modules, 'pythonnet' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    path, clr_loaded, pythonnet_loaded = proc.stdout.split()
    assert Path(path).resolve().is_relative_to(ROOT)
    assert (clr_loaded, pythonnet_loaded) == ("False", "False")
