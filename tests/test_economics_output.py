# tests/test_economics_output.py
"""Tests for economic CSV output files."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _economics_output_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "4",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "simulation.economic.enabled": "true",
        "simulation.economic.rationality": "1.0",
        "simulation.economic.memory.decay": "0.7",
        "economic.fleet.number": "1",
        "economic.fleet.name.fsh0": "Trawlers",
        "economic.fleet.nvessels.fsh0": "5",
        "economic.fleet.homeport.y.fsh0": "1",
        "economic.fleet.homeport.x.fsh0": "1",
        "economic.fleet.gear.fsh0": "bottom_trawl",
        "economic.fleet.max.days.fsh0": "200",
        "economic.fleet.fuel.cost.fsh0": "0.0",
        "economic.fleet.operating.cost.fsh0": "0.0",
        "economic.fleet.target.species.fsh0": "0",
        "economic.fleet.price.sp0.fsh0": "1000.0",
        "economic.fleet.stock.elasticity.sp0.fsh0": "0.0",
    }


class TestEconomicOutput:
    def test_all_output_files_created(self):
        """All economic output files should be created when economics is enabled
        and simulate() receives an output_dir. Exercises the simulate→writer
        wire-up added for the Phase 2 economics-core STATUS-COMPLETE pass."""
        cfg = EngineConfig.from_dict(_economics_output_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            from osmose.engine.output import write_outputs

            out_dir = Path(tmpdir)
            outputs = simulate(cfg, grid, rng, output_dir=out_dir)
            write_outputs(outputs, out_dir, cfg)
            assert len(outputs) == 4

            # Economic CSVs written at end of simulate() (new wire-up)
            assert (out_dir / "econ_effort_Trawlers.csv").exists()
            assert (out_dir / "econ_revenue_Trawlers.csv").exists()
            assert (out_dir / "econ_costs_Trawlers.csv").exists()
            assert (out_dir / "econ_profit_summary.csv").exists()

    def test_no_economic_csvs_when_output_dir_omitted(self):
        """simulate() without output_dir must not write any economic CSVs —
        preserves backward-compat for the 25+ test callers of simulate()
        that never passed an output_dir."""
        cfg = EngineConfig.from_dict(_economics_output_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            simulate(cfg, grid, rng)  # no output_dir kwarg
            assert not (Path(tmpdir) / "econ_profit_summary.csv").exists()

    def test_profit_summary_format(self):
        """Profit summary CSV should have header and one row per fleet."""
        from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
        from osmose.engine.output import write_economic_outputs

        fleet = FleetConfig(
            name="TestFleet",
            n_vessels=2,
            home_port_y=0,
            home_port_x=0,
            gear_type="trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet], grid_ny=2, grid_nx=2, rationality=1.0)
        fs.vessel_revenue[:] = 500.0
        fs.vessel_costs[:] = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            write_economic_outputs(fs, Path(tmpdir))

            profit_file = Path(tmpdir) / "econ_profit_summary.csv"
            assert profit_file.exists()
            with open(profit_file) as f:
                reader = csv.reader(f, delimiter=";")
                header = next(reader)
                assert header == ["fleet", "revenue", "costs", "profit"]
                row = next(reader)
                assert float(row[1]) == pytest.approx(1000.0)  # 2 vessels × 500
                assert float(row[2]) == pytest.approx(200.0)  # 2 vessels × 100
                assert float(row[3]) == pytest.approx(800.0)  # profit

    def test_revenue_costs_csv_created(self):
        """Revenue and costs CSVs should be created per fleet."""
        from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
        from osmose.engine.output import write_economic_outputs

        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=3,
            home_port_y=0,
            home_port_x=0,
            gear_type="trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet], grid_ny=2, grid_nx=2, rationality=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            write_economic_outputs(fs, Path(tmpdir))
            assert (Path(tmpdir) / "econ_effort_Trawlers.csv").exists()
            assert (Path(tmpdir) / "econ_revenue_Trawlers.csv").exists()
            assert (Path(tmpdir) / "econ_costs_Trawlers.csv").exists()
            assert (Path(tmpdir) / "econ_profit_summary.csv").exists()
