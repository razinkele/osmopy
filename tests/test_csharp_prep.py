"""Plan task 1.4: the C# per-step preparation equals the Python engine's, bit for bit (tier E).

A school state is captured from a real Python run, mid-simulation, just before mortality. The C#
engine prepares that step through ``PrepProbe`` (a probe subprocess; the CLR never enters
pytest), and the results must equal, exactly:
- ``_precompute_effective_rates``: starvation, additional, fishing, discard;
- ``compute_feeding_stages``;
- ``compute_school_indices``, both roles;
- ``ResourceState.update(step)``.

The test needs the built engine. It skips without it unless ``OSMOSE_REQUIRE_CSHARP=1``.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.processes.mortality import _precompute_effective_rates
from osmose.engine_cs import csharp_engine_available
from osmose.engine_cs.build import build_bundle
from osmose.engine_cs.bundle import Bundle
from osmose.engine_cs.capabilities import required_features
from tests._csharp_resources import oxygen_coupled, simulation_resources

ROOT = Path(__file__).resolve().parents[1]
CASES = {"eec_full": 30, "bay_of_biscay": 20, "baltic": 129}


class _Stop(Exception):
    pass


def capture_state(demo: str, step: int, workdir: Path, overrides: dict | None = None):
    """(config, grid, raw, state, resources) just before step ``step``'s mortality in a real run;
    ``resources`` is the simulation's own resource biomass at that moment. ``overrides`` are
    config keys set on top of the demo."""
    raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, workdir)["config_file"])))
    raw |= overrides or {}
    raw["output.minimal.enabled"] = "true"
    raw["simulation.time.nyear"] = str(step // 24 + 1)
    config, grid, rng, mv, mo = PythonEngine()._prepare_run(raw, 11)
    got: dict = {}
    real = sim._mortality

    def spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
        if step == target:
            got["state"] = copy.deepcopy(state)
            got["resources"] = resources.biomass.copy()
            raise _Stop
        return real(
            state, resources, config_, rng_, grid_, step=step, species_rngs=species_rngs, ctx=ctx
        )

    target = step
    sim._mortality = spy
    try:
        sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
    except _Stop:
        pass
    finally:
        sim._mortality = real
    return config, grid, raw, got["state"], got["resources"]


def write_state(st, path: Path) -> None:
    b = Bundle(1)
    for name, dt in [
        ("species_id", np.int32), ("age_dt", np.int32), ("length", np.float64), ("weight", np.float64),
        ("abundance", np.float64), ("trophic_level", np.float64), ("cell_x", np.int32),
        ("cell_y", np.int32), ("is_background", np.uint8), ("is_egg", np.uint8),
        ("first_feeding_age_dt", np.int32), ("starvation_rate", np.float64),
    ]:  # fmt: skip
        b.set_array(f"state.{name}", np.ascontiguousarray(getattr(st, name), dtype=dt))
    b.save(path)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> dict:
    ok, why = csharp_engine_available()
    if not ok:
        if os.environ.get("OSMOSE_REQUIRE_CSHARP") == "1":
            pytest.fail(f"OSMOSE_REQUIRE_CSHARP=1 but the C# engine is unavailable: {why}")
        pytest.skip(f"C# engine unavailable ({why}); build it with scripts/build_csharp_engine.sh")
    out = {}
    for demo, step in CASES.items():
        d = tmp_path_factory.mktemp(demo)
        config, grid, raw, st, live = capture_state(demo, step, d)
        build_bundle(config, raw, grid, seed=11).save(d / "model.bin")
        write_state(st, d / "state.bin")
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
            "bridge.load().Diagnostics.PrepProbe.Run(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])"
        )
        args = [
            sys.executable,
            "-c",
            code,
            str(ROOT),
            str(d / "model.bin"),
            str(d / "state.bin"),
            str(step),
            str(d / "prep.bin"),
        ]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr[-4000:]
        out[demo] = (config, grid, st, step, Bundle.load(d / "prep.bin"), raw, d, live)
    return out


@pytest.mark.parametrize("demo", sorted(CASES))
def test_rates_equal_precompute_effective_rates(prepared, demo):
    config, _, st, step, prep, *_ = prepared[demo]
    want = _precompute_effective_rates(st, config, config.mortality_subdt, step)
    names = ("eff_starvation", "eff_additional", "eff_fishing", "discard")
    for name, w in zip(names, want, strict=True):
        assert (
            np.asarray(prep.get(f"prep.{name}")).tobytes()
            == np.asarray(w, dtype=np.float64).tobytes()
        ), name
    assert (want[2] > 0).any() and len(st) > 100  # a real, fished population


@pytest.mark.parametrize("demo", sorted(CASES))
def test_feeding_stages_and_access_indices_equal_the_engine(prepared, demo):
    config, _, st, _, prep, *_ = prepared[demo]
    assert np.array_equal(prep.get("prep.feeding_stage"), compute_feeding_stages(st, config))
    sa = config.stage_accessibility
    for role, name in (("prey", "prep.prey_row"), ("pred", "prep.pred_col")):
        want = sa.compute_school_indices(
            st.species_id, st.age_dt, config.n_dt_per_year, config.all_species_names, role=role
        )
        assert np.array_equal(prep.get(name), want), role


@pytest.mark.parametrize("demo", sorted(CASES))
def test_resource_capacity_equals_the_simulations_own(prepared, demo):
    """Against the resource biomass the running simulation held at this step (not a fresh
    ResourceState, which drops the oxygen coupling, see tests/_csharp_resources.py). Baltic's
    benthos K is scaled by bottom oxygen (feature oxygen_benthos, task 2.5): the C# value equals
    the simulation's there too, and the unscaled product does NOT, so an engine that ignored the
    factor table could not pass."""
    config, grid, _, step, prep, raw, d, live = prepared[demo]
    got = prep.get("prep.resources")[: live.shape[0]]
    rs = simulation_resources(config, grid)
    coupled = oxygen_coupled(config, rs)
    rs.close()
    for r in range(live.shape[0]):
        assert got[r].tobytes() == live[r].tobytes(), (demo, r, coupled)
    assert (coupled is not None) == (demo == "baltic")
    if coupled is not None:
        assert "oxygen_benthos" in required_features(config, raw)
        b = Bundle.load(d / "model.bin")
        frames, fbs, access = (
            b.get(f"resources.{k}") for k in ("forcing", "frame_by_step", "accessibility_by_step")
        )
        unscaled = frames[coupled, fbs[step, coupled]] * access[step, coupled]
        assert unscaled.tobytes() != live[coupled].tobytes()


def test_baltic_background_schools_are_exempt(prepared):
    _, _, st, _, prep, *_ = prepared["baltic"]
    bkg = st.is_background.astype(bool)
    assert bkg.any()
    for name in ("prep.eff_starvation", "prep.eff_additional", "prep.eff_fishing"):
        assert (prep.get(name)[bkg] == 0).all()


def run_probe(model: Path, state: Path, step: int, out: Path) -> Bundle:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from osmose.engine_cs import bridge; "
        "bridge.load().Diagnostics.PrepProbe.Run(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])"
    )
    args = [sys.executable, "-c", code, str(ROOT), str(model), str(state), str(step), str(out)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return Bundle.load(out)


def test_finite_last_stage_thresholds_clamp_and_varying_resource_access(prepared, tmp_path):
    """Every shipped config ends its stage thresholds at +inf (so the clamp to the last stage
    never fires) and has constant resource accessibility. Give EEC finite thresholds and a
    time-varying accessibility file."""
    config, grid, st, step, _, raw, d, _ = prepared["eec_full"]
    cfg = copy.copy(config)
    sa = copy.deepcopy(config.stage_accessibility)
    for role in ("prey", "pred"):
        for stages in sa._stages_by_role[role].values():
            t = np.asarray(stages.thresholds, dtype=np.float64).copy()
            t[-1] = (
                1.0 if len(t) == 1 else t[-2] + 1.0
            )  # finite: old schools fall past every threshold
            stages.thresholds = t
    cfg.stage_accessibility = sa
    rsc_key = next(
        k
        for k, v in raw.items()
        if k.startswith("species.type.sp") and v.strip().lower() == "resource"
    )
    access_csv = tmp_path / "access.csv"
    access_csv.write_text("step;value\n" + "".join(f"{i};{0.1 + 0.02 * i}\n" for i in range(24)))
    cfg.raw_config = dict(config.raw_config) | {
        rsc_key.replace("species.type.", "species.accessibility2fish.file."): str(access_csv)
    }
    build_bundle(cfg, raw, grid, seed=11).save(tmp_path / "model.bin")
    prep = run_probe(tmp_path / "model.bin", d / "state.bin", step, tmp_path / "prep.bin")
    for role, name in (("prey", "prep.prey_row"), ("pred", "prep.pred_col")):
        want = sa.compute_school_indices(
            st.species_id, st.age_dt, cfg.n_dt_per_year, cfg.all_species_names, role=role
        )
        assert np.array_equal(prep.get(name), want), role
    rs = simulation_resources(cfg, grid)
    assert oxygen_coupled(cfg, rs) is None
    rs.update(step)
    assert prep.get("prep.resources")[: rs.n_resources].tobytes() == rs.biomass.tobytes()
    assert any(s.accessibility_ts is not None for s in rs.species)
    rs.close()
