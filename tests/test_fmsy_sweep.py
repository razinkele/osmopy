"""Tests for osmose.validation.fmsy_sweep — mode detection + species->fishery map."""

import json

import numpy as np
import pandas as pd
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.validation.fmsy_sweep import SweepPoint, derive_reference_points, fishing_override

BALTIC = "data/baltic/baltic_all-parameters.csv"


def _baltic_cfg() -> dict[str, str]:
    raw = dict(OsmoseConfigReader().read(BALTIC))
    raw["simulation.time.nyear"] = "2"
    return raw


def test_fisheries_mode_override_actually_changes_fishing_rate():
    raw = _baltic_cfg()
    cfg = EngineConfig.from_dict(dict(raw))
    key, base = fishing_override(raw, cfg, 0)
    assert key.startswith("fisheries.rate.base.fsh")  # baltic is v4 fisheries-mode
    assert base == pytest.approx(cfg.fishing_rate[0])
    # overriding the returned key MUST move fishing_rate[0] (the no-op-trap guard)
    bumped = dict(raw)
    bumped[key] = "9.0"
    assert EngineConfig.from_dict(bumped).fishing_rate[0] == pytest.approx(9.0)
    # and ONLY species 0 if 1:1 (baltic is 1:1)
    assert EngineConfig.from_dict(bumped).fishing_rate[1] == pytest.approx(cfg.fishing_rate[1])


def test_legacy_mode_override():
    raw = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Fish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "1",
        "mortality.fishing.rate.method.sp0": "constant",
        "mortality.fishing.rate.sp0": "0.2",
    }
    cfg = EngineConfig.from_dict(dict(raw))
    key, base = fishing_override(raw, cfg, 0)
    assert key == "mortality.fishing.rate.sp0"
    bumped = dict(raw)
    bumped[key] = "0.9"
    assert EngineConfig.from_dict(bumped).fishing_rate[0] == pytest.approx(0.9)


def _curve(species, fs, yields, ssbs, frealized=None):
    fr = frealized or fs
    return [SweepPoint(species, fn, r, y, s) for fn, r, y, s in zip(fs, fr, yields, ssbs)]


def test_single_peak_fmsy_bmsy_b0_blim():
    # yield rises then falls; peak at f_nominal=0.6 (realized 0.5); SSB declines with F
    fs = [0.0, 0.3, 0.6, 0.9, 1.2]
    rs = [0.0, 0.25, 0.5, 0.75, 1.0]
    yields = [0.0, 8.0, 10.0, 7.0, 3.0]
    ssbs = [1000.0, 700.0, 500.0, 300.0, 150.0]
    rp = derive_reference_points({"cod": _curve("cod", fs, yields, ssbs, rs)})["cod"]
    assert rp.fmsy == pytest.approx(0.5)  # realized F at the yield peak
    assert rp.bmsy == pytest.approx(500.0)  # SSB at the peak
    assert rp.b0 == pytest.approx(1000.0)  # SSB at F=0
    assert rp.blim == pytest.approx(200.0)  # 0.2 * B0
    assert not rp.fmsy_at_boundary and not rp.multi_peak


def test_monotone_increasing_is_boundary():
    fs = [0.0, 0.5, 1.0]
    rp = derive_reference_points({"x": _curve("x", fs, [0.0, 5.0, 9.0], [900.0, 500.0, 200.0])})[
        "x"
    ]
    assert rp.fmsy_at_boundary and any("boundary" in c.lower() for c in rp.caveats)


def test_monotone_decreasing_no_fmsy():
    fs = [0.0, 0.5, 1.0]
    rp = derive_reference_points({"x": _curve("x", fs, [9.0, 5.0, 1.0], [900.0, 500.0, 200.0])})[
        "x"
    ]
    assert rp.fmsy is None and any("no" in c.lower() for c in rp.caveats)


def test_two_peaks_flagged():
    fs = [0.0, 0.25, 0.5, 0.75, 1.0]
    rp = derive_reference_points(
        {"x": _curve("x", fs, [0, 9, 2, 8, 1], [900, 700, 500, 300, 150])}
    )["x"]
    assert rp.multi_peak


def test_b0_nonpositive_no_blim():
    fs = [0.0, 0.5]
    rp = derive_reference_points({"x": _curve("x", fs, [0.0, 5.0], [0.0, -1.0])})["x"]
    assert rp.blim is None


# ---------------------------------------------------------------------------
# Task 3: engine sweep tests
# ---------------------------------------------------------------------------


def _tiny_fished_legacy_cfg() -> dict[str, str]:
    """Legacy-mode single-species config (nfisheries unset), ndt=12, 6 yr, seeded."""
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "6",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "20",
        "species.name.sp0": "Fish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.5",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "4",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "1",
        "mortality.fishing.rate.method.sp0": "constant",
        "mortality.fishing.rate.sp0": "0.3",
        "population.seeding.biomass.sp0": "100.0",
    }


@pytest.mark.slow
def test_sweep_end_to_end_tiny_legacy():
    from osmose.validation.fmsy_sweep import compute_model_reference_points

    refs = compute_model_reference_points(
        _tiny_fished_legacy_cfg(),
        grid=np.array([0.0, 0.4, 0.8, 1.2]),
        n_years=6,
        replicates=1,
        window_years=2,
        # Serial in-process path (workers<=1): the sweep's ProcessPoolExecutor uses a
        # spawn context (correct for production's fork-after-numba-threads case) but
        # spawning a pool from inside a pytest-xdist worker deadlocks. This end-to-end
        # test only needs the sweep logic, which the serial path exercises identically.
        max_workers=1,
    )
    rp = refs["Fish"]
    assert rp.b0 is not None and rp.b0 > 0  # F=0 has the largest (unfished) SSB
    assert any(p.yield_eq > 0 for p in rp.curve)  # yield reader + forced output worked


def test_sweep_assembles_curves_stubbed(monkeypatch):
    """Fast default-suite coverage: stub the engine so no real run happens; assert the runner
    forces the SSB flag, applies fishing_override, and assembles a curve."""
    import osmose.validation.fmsy_sweep as sweep

    seen_cfgs = []

    class _FakeRes:
        def yield_biomass(self):
            return pd.DataFrame({"Time": [0.0, 1.0], "Fish": [5.0, 5.0]})

        def ssb(self):
            return pd.DataFrame({"Time": [0.0, 1.0], "Fish": [100.0, 100.0]})

        def mortality(self, sp):
            return pd.DataFrame({"Time": [0.0, 1.0], "Fishing": [0.3, 0.3], "species": [sp, sp]})

    def _fake_run(self, cfg, seed=0, **kw):
        seen_cfgs.append(cfg)
        return _FakeRes()

    monkeypatch.setattr(sweep.PythonEngine, "run_in_memory", _fake_run)
    from osmose.validation.fmsy_sweep import compute_model_reference_points

    refs = compute_model_reference_points(
        _tiny_fished_legacy_cfg(),
        grid=np.array([0.0, 0.5]),
        n_years=4,
        replicates=1,
        window_years=1,
        max_workers=1,
    )
    assert "Fish" in refs
    assert all(c.get("output.ssb.enabled") == "true" for c in seen_cfgs)  # forced output
    assert len(refs["Fish"].curve) == 2  # one SweepPoint per grid F


# ---------------------------------------------------------------------------
# Task 4: CLI + sidecar tests
# ---------------------------------------------------------------------------


def test_write_model_sidecar(tmp_path):
    from osmose.validation.fmsy_sweep import ModelReferencePoint
    from scripts.compute_model_reference_points import write_model_sidecar

    refs = {
        "cod": ModelReferencePoint(
            "cod",
            fmsy=0.3,
            bmsy=118000.0,
            b0=410000.0,
            blim=82000.0,
            fmsy_at_boundary=False,
            multi_peak=False,
        )
    }
    out = tmp_path / "fisheries_model_reference_points.json"
    write_model_sidecar(refs, out, meta={"grid": [0.0, 1.0], "replicates": 3})
    d = json.loads(out.read_text())
    assert d["cod"]["fmsy"] == 0.3 and d["cod"]["blim"] == 82000.0
    assert d["_meta"]["replicates"] == 3
