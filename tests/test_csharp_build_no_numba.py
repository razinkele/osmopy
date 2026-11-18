"""A process that runs the C# engine never imports Numba (task 1.11's memory budget).

Importing Numba and LLVM costs a process about 64 MB, and the C# path used none of it: the bundle
builder imported plain helpers from ``processes.mortality`` / ``processes.movement``, which load
Numba at import time. The helpers now live in the Numba-free ``processes.precompute``. Spec 10.4's
budget, "peak memory per run no more than the Python engine's", rests on that.

Each check runs in a fresh interpreter: this pytest process has Numba loaded already.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osmose.engine_cs import csharp_engine_available

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "data" / "eec_full" / "eec_all-parameters.csv"  # maps and resources: both helpers

_PROBE = """
import json, sys
sys.path.insert(0, {root!r})
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine_cs.build import allocate_results, build_bundle
from osmose.engine.simulate import StepOutput  # noqa: F401  (what CSharpEngine's outputs use)
import osmose.results  # noqa: F401
raw = dict(OsmoseConfigReader().read({cfg!r}))
raw["simulation.time.nyear"] = "1"
raw["output.minimal.enabled"] = "true"
raw.update(json.loads(sys.argv[2]))
cfg = EngineConfig.from_dict(raw)
bundle = build_bundle(cfg, raw, PythonEngine()._resolve_grid(raw), seed=1)
allocate_results(bundle, cfg)
if sys.argv[1] == "run":
    from osmose.engine_cs import CSharpEngine
    CSharpEngine().run_in_memory(raw, seed=1).biomass()
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("numba", "llvmlite"))
print(json.dumps({{"loaded": loaded[:5], "n_steps": int(cfg.n_steps)}}))
"""


# A salinity gate and an oxygen coupling, on constant fields: the builder then also runs the
# engine's weight and factor helpers, which must come from Numba-free modules too.
GATES = {
    "movement.salinity.gate.enabled": "true",
    "movement.salinity.field.constant": "4.5",
    "movement.salinity.gate.species.enabled.sp0": "true",
    "ltl.oxygen.benthos.enabled": "true",
    "ltl.oxygen.benthos.rsc": "Mesozoo",
    "oxygen.value": "45.0",
}


def _probe(tmp_path: Path, mode: str, overrides: dict | None = None) -> dict:
    script = tmp_path / "probe.py"
    script.write_text(_PROBE.format(root=str(ROOT), cfg=str(CFG)))
    proc = subprocess.run(
        [sys.executable, str(script), mode, json.dumps(overrides or {})],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("overrides", [{}, GATES], ids=["plain", "salinity gate and oxygen"])
def test_the_bundle_builder_does_not_import_numba(overrides, tmp_path):
    out = _probe(tmp_path, "build", overrides)
    assert out["n_steps"] == 24  # the probe really built a bundle
    assert out["loaded"] == [], f"the C# build path imported {out['loaded']}"


def test_a_csharp_run_does_not_import_numba(tmp_path):
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    out = _probe(tmp_path, "run")
    assert out["loaded"] == [], f"a C# run imported {out['loaded']}"


def test_csharp_calibration_workers_skip_numbas_thread_rule(monkeypatch, tmp_path):
    """Applying the rule imports Numba; a C# worker runs its own threads."""
    from osmose.calibration import problem as P

    calls: list[int] = []
    monkeypatch.setattr(P, "apply_worker_threads", calls.append)
    monkeypatch.setattr(P, "OsmoseCalibrationProblem", lambda **kw: kw)  # not built here
    monkeypatch.setattr(P, "_WORKER_PROBLEM", None)  # restored after the test
    spec = P._EvalSpec(
        free_params=[], objective_fns=[], base_config_path=tmp_path, work_dir=tmp_path,
        jar_path=None, java_cmd="java", enable_cache=False, cache_dir=None,
        subprocess_timeout=1, cleanup_after_eval=False, use_java_engine=False,
        use_registry=False, n_workers=3, engine="csharp",
    )  # fmt: skip
    P._worker_init(spec)
    assert calls == []
    P._worker_init(dataclasses.replace(spec, engine="python"))
    assert calls == [3]
