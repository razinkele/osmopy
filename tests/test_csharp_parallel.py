"""Plan task 1.9: parallel runs are deterministic, and a one-thread step allocates nothing.

- The result buffers are byte-identical at 1, 2 and 14 threads, on EEC and Bay of Biscay
  (mortality runs cells heaviest first across workers, movement runs chunks: spec D4).
- Two runs with one seed are byte-identical; two seeds differ.
- A steady-state step at one thread allocates no managed memory.

Runs go through ``PipelineProbe`` in a child process (the CLR never enters pytest). The test
needs the built engine; it skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    "out.record_step",
    "out.window_steps",
    "out.biomass",
    "out.abundance",
    "out.mortality",
    "out.yield",
)


@pytest.fixture(scope="module", autouse=True)
def engine_ok():
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


def _model(demo: str, d: Path, nyear: int, *, seed: int, threads: int, name: str) -> Path:
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, d)["config_file"])))
    raw |= {"output.minimal.enabled": "true", "simulation.time.nyear": str(nyear)}
    config, grid, *_ = PythonEngine()._prepare_run(raw, 0)
    path = d / name
    build_bundle(config, raw, grid, seed=seed, threads=threads).save(path)
    return path


def _probe(method: str, *args) -> Bundle:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        f"f = bridge.load().Diagnostics.PipelineProbe.{method}; "
        "a = [int(x) if x.isdigit() else x for x in sys.argv[2:]]; f(*a)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(args[-1])


def _results(b: Bundle) -> bytes:
    return b"".join(np.ascontiguousarray(b.get(name)).tobytes() for name in RESULTS)


@pytest.mark.parametrize(("demo", "nyear"), [("eec_full", 2), ("bay_of_biscay", 1)])
def test_results_are_byte_identical_at_1_2_and_14_threads(demo, nyear, tmp_path):
    runs = {}
    for threads in (1, 2, 14):
        model = _model(demo, tmp_path, nyear, seed=21, threads=threads, name=f"m{threads}.bin")
        runs[threads] = _probe("Run", model, tmp_path / f"o{threads}.bin")
    status, _, _, peak = runs[1].get("out.summary")
    assert status == 0 and peak > 100  # a real run, with cells worth spreading
    assert _results(runs[1]) == _results(runs[2]) == _results(runs[14])
    assert np.asarray(runs[1].get("out.mortality")).sum() > 0


def test_repeat_runs_are_identical_and_seeds_differ(tmp_path):
    a = _probe(
        "Run",
        _model("eec_full", tmp_path, 1, seed=5, threads=2, name="a.bin"),
        tmp_path / "a_out.bin",
    )
    b = _probe(
        "Run",
        _model("eec_full", tmp_path, 1, seed=5, threads=2, name="b.bin"),
        tmp_path / "b_out.bin",
    )
    c = _probe(
        "Run",
        _model("eec_full", tmp_path, 1, seed=6, threads=2, name="c.bin"),
        tmp_path / "c_out.bin",
    )
    assert _results(a) == _results(b)
    assert _results(a) != _results(c)


def test_a_steady_state_step_allocates_nothing_at_one_thread(tmp_path):
    model = _model("eec_full", tmp_path, 6, seed=3, threads=1, name="m.bin")
    allocated, cap_before, cap_after, schools = _probe(
        "Allocations", model, 100, 24, tmp_path / "a.bin"
    ).get("out.allocations")
    assert schools > 100
    assert cap_before == cap_after, "the store grew inside the window: choose a later one"
    assert allocated == 0, f"{allocated} bytes allocated over 24 steady-state steps"
