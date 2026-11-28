"""The fleet-rate recalibration driver: targets, fleet set, evaluator wiring, CSV apply (no engine)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.recalibrate_fleet_rates import (
    FLEETS,
    apply_rates,
    load_catch_targets,
    make_evaluator,
)

REPO = Path(__file__).resolve().parents[1]
TARGETS = REPO / "data" / "baltic" / "reference" / "biomass_targets.csv"
FISHING = REPO / "data" / "baltic" / "baltic_param-fishing.csv"


def test_load_catch_targets():
    t = load_catch_targets(TARGETS)
    assert set(t) == {"cod_west", "cod_east", "herring", "sprat", "flounder"}
    assert t["cod_west"] == pytest.approx(1428, rel=1e-3)
    assert t["flounder"] == pytest.approx(9520, rel=1e-3)


def test_fleets_cover_exactly_four():
    assert set(FLEETS) == {"fsh0", "fsh8", "fsh1", "fsh3"}  # sprat fsh2 keeps its ICES-F anchor
    assert FLEETS["fsh0"] == ("cod_west", "fisheries.rate.base.fsh0")
    assert FLEETS["fsh3"][0] == "flounder"


class _FakeResult:
    def __init__(self, cfg):
        self.cfg = cfg

    def yield_biomass(self):
        rate = float(self.cfg["fisheries.rate.base.fsh0"])
        return pd.DataFrame({"Time": range(6), "cod_west": [rate * 1000.0] * 6})


class _FakeEngine:
    def __init__(self):
        self.configs: list[dict] = []

    def run_in_memory(self, cfg, seed=0):
        self.configs.append(dict(cfg))
        return _FakeResult(cfg)


def test_make_evaluator_sets_only_its_rate():
    base = {"fisheries.rate.base.fsh0": "0.039", "fisheries.rate.base.fsh1": "0.38", "x": "1"}
    eng = _FakeEngine()
    ev = make_evaluator(base, "fisheries.rate.base.fsh0", "cod_west", eng, seeds=[1, 2])
    assert ev(0.5) == pytest.approx(500.0)
    assert len(eng.configs) == 2  # one run per seed
    for cfg in eng.configs:
        diff = {k for k in cfg if cfg[k] != base.get(k)}
        assert diff == {"fisheries.rate.base.fsh0", "simulation.time.nyear"}
        assert float(cfg["fisheries.rate.base.fsh0"]) == 0.5
    assert base["fisheries.rate.base.fsh0"] == "0.039"  # base not mutated


def test_apply_rates_replaces_in_place(tmp_path):
    p = tmp_path / "fishing.csv"
    shutil.copy(FISHING, p)
    before = p.read_text().splitlines()
    apply_rates(p, {"fsh0": 0.5})
    after = p.read_text().splitlines()
    rate_lines = [ln for ln in after if ln.startswith("fisheries.rate.base.fsh0;")]
    assert rate_lines == ["fisheries.rate.base.fsh0;0.5"]  # replaced, never appended
    assert len(after) == len(before)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1


def test_apply_rates_refuses_a_missing_key(tmp_path):
    p = tmp_path / "fishing.csv"
    p.write_text("fisheries.rate.base.fsh1;0.38\n")
    with pytest.raises(KeyError):
        apply_rates(p, {"fsh0": 0.5})
