"""Smoke tests for scripts/run_dsvm_demo.py.

The full eec_full demo run takes ~1.5s; this test gates the demo's
config-override + output-reading helpers via a tiny synthetic config
that runs in well under a second.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = PROJECT_ROOT / "scripts" / "run_dsvm_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("run_dsvm_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_dsvm_overrides_no_op_when_disabled():
    demo = _load_demo_module()
    raw = {"simulation.time.nyear": "2"}
    overrides = demo._build_dsvm_overrides(raw, enable_dsvm=False)
    assert overrides == raw  # no-op
    assert "simulation.economic.enabled" not in overrides


def test_build_dsvm_overrides_adds_required_keys():
    demo = _load_demo_module()
    raw = {"simulation.time.nyear": "2"}
    overrides = demo._build_dsvm_overrides(raw, enable_dsvm=True)
    # Top-level economic gating keys
    assert overrides["simulation.economic.enabled"] == "true"
    assert overrides["economic.fleet.number"] == "2"
    # Two fleets configured
    assert overrides["economic.fleet.name.fsh0"] == "DemersalTrawlers"
    assert overrides["economic.fleet.name.fsh1"] == "PelagicTrawlers"
    # Per-species pricing keys present for both fleets across all 14 species
    for fid in (0, 1):
        for sp in range(14):
            assert f"economic.fleet.price.sp{sp}.fsh{fid}" in overrides
            assert f"economic.fleet.stock.elasticity.sp{sp}.fsh{fid}" in overrides
    # Spatial output enabled (required for fleet effort maps)
    assert overrides["output.spatial.enabled"] == "true"


def test_demersal_targets_have_unit_elasticity():
    demo = _load_demo_module()
    raw = {"simulation.time.nyear": "2"}
    overrides = demo._build_dsvm_overrides(raw, enable_dsvm=True)
    # DEMERSAL_TARGETS = [5, 3, 7, 8] (cod, whiting, sole, plaice)
    for sp in demo.DEMERSAL_TARGETS:
        assert overrides[f"economic.fleet.stock.elasticity.sp{sp}.fsh0"] == "1.0"
    # Non-targets: zero elasticity
    for sp in [0, 1, 2, 4, 6]:
        assert overrides[f"economic.fleet.stock.elasticity.sp{sp}.fsh0"] == "0.0"


def test_pelagic_targets_have_unit_elasticity():
    demo = _load_demo_module()
    raw = {"simulation.time.nyear": "2"}
    overrides = demo._build_dsvm_overrides(raw, enable_dsvm=True)
    # PELAGIC_TARGETS = [11, 12, 10] (herring, sardine, mackerel)
    for sp in demo.PELAGIC_TARGETS:
        assert overrides[f"economic.fleet.stock.elasticity.sp{sp}.fsh1"] == "1.0"


def test_read_econ_outputs_returns_empty_when_no_csvs(tmp_path):
    demo = _load_demo_module()
    summary = demo._read_econ_outputs(tmp_path)
    assert summary == {"fleets": {}}


def test_read_econ_outputs_parses_semicolon_csv(tmp_path):
    """Regression test for the pd.read_csv default-delimiter bug:
    economics writer emits semicolon-delimited CSVs without a header,
    and the reader must pass sep=';' header=None.
    """
    demo = _load_demo_module()
    # Mock the writer's output format: per-vessel revenue across one period.
    (tmp_path / "econ_revenue_TestFleet.csv").write_text(
        "1000.0;2000.0;3000.0;4000.0\n"
    )
    (tmp_path / "econ_costs_TestFleet.csv").write_text(
        "100.0;200.0;300.0;400.0\n"
    )
    summary = demo._read_econ_outputs(tmp_path)
    assert "TestFleet" in summary["fleets"]
    fleet = summary["fleets"]["TestFleet"]
    assert fleet["total_revenue_eur"] == 10_000.0  # sum of 1k+2k+3k+4k
    assert fleet["total_costs_eur"] == 1_000.0
    assert fleet["net_profit_eur"] == 9_000.0
