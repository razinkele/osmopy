"""Plan task 2.3, end to end: recruitment regulation through the whole C# engine.

The stage itself is checked bit for bit against the Python engine's ``reproduction`` in
tests/test_csharp_life_cycle.py (each curve, both seeding modes, the RV gate, Baltic's own
curves and series). Here the features run inside real runs, on EEC with:

- ``curves``: a Shepherd curve on every species (config keys, as a user would write them);
- ``all``: the same, plus the linear seeding mode and an RV gate on two species.

Runs are byte-identical at 1, 2 and 14 threads, and the three configurations (plain, curves, all)
give three different results: each feature reaches the output. The runs happen in child
processes and need the built engine.
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
from osmose.engine_cs.capabilities import csharp_engine_block_reason, required_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ("out.biomass", "out.abundance", "out.mortality", "out.yield")
N_FOCAL = 14


def _curves() -> dict[str, str]:
    keys: dict[str, str] = {}
    for sp in range(N_FOCAL):
        keys[f"stock.recruitment.type.sp{sp}"] = "shepherd"
        keys[f"stock.recruitment.ssbhalf.sp{sp}"] = "5000.0"
        keys[f"stock.recruitment.shape.sp{sp}"] = "1.6"
    return keys


def _bundle(d: Path, kind: str, threads: int = 1) -> tuple[Bundle, list[str]]:
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo("eec_full", d)["config_file"])))
    raw |= {"output.minimal.enabled": "true", "simulation.time.nyear": "2"}
    if kind != "plain":
        raw |= _curves()
    if kind == "all":
        raw["population.seeding.mode"] = "linear"
    config, grid, *_ = PythonEngine()._prepare_run(raw, 0)
    if kind == "all":  # an RV gate on two species; the series comes from a file in real configs
        enabled = np.zeros(config.n_species, dtype=bool)
        enabled[[2, 3]] = True
        config.rv_gate_enabled = enabled
        config.rv_gate_factor_by_index = np.array([0.5, 0.25])
        config.rv_gate_offset = 0
    assert csharp_engine_block_reason(raw) is None
    return build_bundle(config, raw, grid, seed=9, threads=threads), required_features(config, raw)


def _run(b: Bundle, d: Path, name: str) -> bytes:
    b.save(d / f"{name}.bin")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.PipelineProbe.Run(sys.argv[2], sys.argv[3])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(d / f"{name}.bin"), str(d / f"{name}.out")],
        capture_output=True, text=True, timeout=1200, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = Bundle.load(d / f"{name}.out")
    assert out.get("out.summary")[0] == 0  # completed
    return b"".join(np.ascontiguousarray(out.get(name_)).tobytes() for name_ in RESULTS)


def test_the_configs_need_exactly_these_features(tmp_path):
    assert _bundle(tmp_path, "plain")[1] == ["core"]
    assert _bundle(tmp_path, "curves")[1] == ["core", "stock_recruitment"]
    b, features = _bundle(tmp_path, "all")
    assert features == ["core", "stock_recruitment", "rv_gate", "seeding_linear"]
    assert b.get("reproduction.sr.type").tolist() == [4] * N_FOCAL
    assert b.get("reproduction.seeding_mode") == 1
    assert b.get("reproduction.rv.factor_by_year")[:, [2, 3]].tolist() == [[0.5, 0.5], [0.25, 0.25]]
    plain = _bundle(tmp_path, "plain")[0]
    assert not [n for n in plain.names() if n.startswith(("reproduction.sr.s", "reproduction.rv."))]


def test_recruitment_runs_are_deterministic_and_each_feature_reaches_the_output(tmp_path):
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    full = {t: _run(_bundle(tmp_path, "all", t)[0], tmp_path, f"all{t}") for t in (1, 2, 14)}
    assert full[1] == full[2] == full[14]
    plain = _run(_bundle(tmp_path, "plain")[0], tmp_path, "plain")
    curves = _run(_bundle(tmp_path, "curves")[0], tmp_path, "curves")
    assert len({plain, curves, full[1]}) == 3
