"""Plan task 2.5: bottom oxygen scaling a resource's carrying capacity (feature oxygen_benthos).

The config under test needs exactly core plus this feature: EEC with minimal outputs, the
coupling switched on for one of its resources, and a synthetic oxygen forcing on a grid of a
DIFFERENT shape than the model's, so ``ResourceState``'s regrid path is in play.

- The bundle's factor table reproduces ``ResourceState.update`` bit for bit, for a regridded
  file and for a constant field, and the product without the table does not.
- Nothing is coupled without a forcing or a resource of the configured name, as in Python.
- A forcing whose frame count is not ``n_dt_per_year`` is refused by the builder.
- The C# engine's own ``UpdateResources`` equals the running simulation's resource biomass.
- Runs with the feature are byte-identical at 1, 2 and 14 threads, and differ from the same
  run without it.

Baltic's real coupling is covered in tests/test_csharp_build.py and tests/test_csharp_prep.py.
The CLR-hosting checks run in child processes and need the built engine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import csharp_engine_block_reason, required_features
from tests._csharp_resources import oxygen_coupled, simulation_resources
from tests.test_csharp_prep import capture_state, run_probe, write_state

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = "Mesozoo"  # an EEC resource the fish feed on
RESULTS = ("out.biomass", "out.abundance", "out.mortality", "out.yield")
ENTRIES = ("coupled", "factor", "frame_by_step")


def _oxygen_file(path: Path, frames: int = 24, shape: tuple[int, int] = (11, 15)) -> Path:
    """Oxygen 0..200 (the Hill factor then spans 0 to ~0.97), per frame and per cell distinct."""
    data = np.random.default_rng(3).uniform(0.0, 200.0, (frames, *shape))
    ds = xr.Dataset({"o2b": (["time", "y", "x"], data)})
    ds.to_netcdf(path)
    ds.close()
    return path


def _coupling(**extra: str) -> dict[str, str]:
    return {"ltl.oxygen.benthos.enabled": "true", "ltl.oxygen.benthos.rsc": RESOURCE, **extra}


def _file(nc: Path) -> dict[str, str]:
    return _coupling(**{"oxygen.filename": str(nc), "oxygen.varname": "o2b"})


def _case(d: Path, overrides: dict[str, str], nyear: int = 2, seed: int = 4, threads: int = 1):
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo("eec_full", d)["config_file"])))
    raw |= overrides | {"output.minimal.enabled": "true", "simulation.time.nyear": str(nyear)}
    config, grid, *_ = PythonEngine()._prepare_run(raw, 0)
    return config, raw, grid, build_bundle(config, raw, grid, seed=seed, threads=threads)


def _bundle_k(b: Bundle, step: int, r: int, table: bool = True) -> np.ndarray:
    """The C# engine's K for resource ``r``: (forcing * access) * factor, as it computes it."""
    k = b.get("resources.forcing")[r, b.get("resources.frame_by_step")[step, r]]
    k = k * b.get("resources.accessibility_by_step")[step, r]
    if table and b.get("resources.oxygen.coupled")[r]:
        k = k * b.get("resources.oxygen.factor")[b.get("resources.oxygen.frame_by_step")[step]]
    return k


def _need_engine() -> None:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")


# ------------------------------------------------------------------ the builder (no CLR)


@pytest.mark.parametrize("mode", ["regridded file", "constant"])
def test_the_factor_table_reproduces_resource_state_update(mode, tmp_path):
    overrides = (
        _file(_oxygen_file(tmp_path / "o2.nc"))
        if mode == "regridded file"
        else _coupling(**{"oxygen.value": "45.0"})
    )
    config, raw, grid, b = _case(tmp_path, overrides)
    assert required_features(config, raw) == ["core", "oxygen_benthos"]
    assert csharp_engine_block_reason(raw) is None  # core plus an implemented feature: it runs
    rs = simulation_resources(config, grid)
    coupled = oxygen_coupled(config, rs)
    assert coupled is not None and rs.species[coupled].name == RESOURCE
    assert b.get("resources.oxygen.coupled").tolist() == [
        int(r == coupled) for r in range(rs.n_resources)
    ]
    factor = b.get("resources.oxygen.factor")
    assert factor.shape == ((24 if mode == "regridded file" else 1), grid.ny * grid.nx)
    assert ((factor >= 0) & (factor <= 1)).all() and factor.min() < 0.9  # it really scales
    for step in (0, 1, 23, 24, 47):
        rs.update(step)
        for r in range(rs.n_resources):
            assert _bundle_k(b, step, r).tobytes() == rs.biomass[r].tobytes(), (step, r)
        # Without the table the coupled resource is wrong: the table is what carries it.
        assert _bundle_k(b, step, coupled, table=False).tobytes() != rs.biomass[coupled].tobytes()
    rs.close()
    if mode == "regridded file":  # frames differ, so the step-to-frame map matters
        assert b.get("resources.oxygen.frame_by_step").tolist() == [s % 24 for s in range(48)]
        assert not np.array_equal(factor[0], factor[1])


@pytest.mark.parametrize("why", ["no forcing", "no resource of that name"])
def test_nothing_is_coupled_without_a_forcing_or_a_matching_resource(why, tmp_path):
    overrides = (
        _coupling()
        if why == "no forcing"
        else _file(_oxygen_file(tmp_path / "o2.nc")) | {"ltl.oxygen.benthos.rsc": "NoSuchResource"}
    )
    config, raw, grid, b = _case(tmp_path, overrides)
    assert "oxygen_benthos" in required_features(config, raw)  # declared, so the entries exist
    assert not b.get("resources.oxygen.coupled").any()
    rs = simulation_resources(config, grid)
    for step in (0, 30):
        rs.update(step)
        for r in range(rs.n_resources):
            assert _bundle_k(b, step, r).tobytes() == rs.biomass[r].tobytes(), (step, r)
    rs.close()


def test_a_forcing_with_the_wrong_frame_count_is_refused_by_the_builder(tmp_path):
    """PhysicalData indexes step % frames, so 12 frames on a 24-step year would misalign
    silently; simulate() raises (_load_oxygen_data) and so does the bundle builder."""
    nc = _oxygen_file(tmp_path / "o2_12.nc", frames=12)
    with pytest.raises(ValueError, match=r"12 frame.*ndtperyear=24"):
        _case(tmp_path, _file(nc))


def test_a_config_without_the_feature_carries_no_oxygen_entries(tmp_path):
    config, raw, _, b = _case(tmp_path, {})
    assert required_features(config, raw) == ["core"]
    assert not [n for n in b.names() if n.startswith("resources.oxygen.")]


# ------------------------------------------------------------------ the engine (child processes)


def test_the_engine_computes_the_simulations_resource_biomass(tmp_path):
    """C#'s own UpdateResources on the bundle, against the resource biomass the running Python
    simulation held at that step: equal for every resource, the coupled one included."""
    _need_engine()
    step = 30
    overrides = _file(_oxygen_file(tmp_path / "o2.nc"))
    config, grid, raw, state, live = capture_state("eec_full", step, tmp_path, overrides)
    b = build_bundle(config, raw, grid, seed=11)
    b.save(tmp_path / "model.bin")
    write_state(state, tmp_path / "state.bin")
    prep = run_probe(tmp_path / "model.bin", tmp_path / "state.bin", step, tmp_path / "prep.bin")
    got = prep.get("prep.resources")[: live.shape[0]]
    rs = simulation_resources(config, grid)
    coupled = oxygen_coupled(config, rs)
    rs.close()
    assert coupled is not None
    for r in range(live.shape[0]):
        assert got[r].tobytes() == live[r].tobytes(), r
    assert _bundle_k(b, step, coupled, table=False).tobytes() != live[coupled].tobytes()


def _run(model: Path, out: Path) -> bytes:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.PipelineProbe.Run(sys.argv[2], sys.argv[3])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(model), str(out)],
        capture_output=True, text=True, timeout=1200, cwd=ROOT,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-4000:]
    b = Bundle.load(out)
    assert b.get("out.summary")[0] == 0  # completed
    return b"".join(np.ascontiguousarray(b.get(name)).tobytes() for name in RESULTS)


def test_runs_with_the_feature_are_deterministic_and_differ_from_runs_without(tmp_path):
    _need_engine()
    overrides = _file(_oxygen_file(tmp_path / "o2.nc"))
    results = {}
    for threads in (1, 2, 14):
        *_, b = _case(tmp_path, overrides, threads=threads)
        b.save(tmp_path / f"on{threads}.bin")
        results[threads] = _run(tmp_path / f"on{threads}.bin", tmp_path / f"on{threads}.out")
    assert results[1] == results[2] == results[14]
    *_, plain = _case(tmp_path, {})
    plain.save(tmp_path / "off.bin")
    assert _run(tmp_path / "off.bin", tmp_path / "off.out") != results[1]
