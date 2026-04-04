"""Tests for complete bioenergetic outputs (all 5 bioen CSVs)."""
from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.output import write_outputs
from osmose.engine.processes.energy_budget import compute_energy_budget
from osmose.engine.simulate import simulate

from tests.test_engine_bioen_integration import _make_bioen_config


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run_bioen_sim(cfg: dict[str, str], tmp_path):
    """Run a short bioen simulation and write outputs. Returns (outputs, config)."""
    config = EngineConfig.from_dict(cfg)
    grid = Grid.from_dimensions(ny=5, nx=5)
    rng = np.random.default_rng(42)
    outputs = simulate(config, grid, rng)
    write_outputs(outputs, tmp_path, config, prefix="osmose")
    return outputs, config


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestBioenOutputsComplete:
    def test_all_bioen_csvs_created(self, tmp_path):
        """All 5 bioen CSVs are created when all output flags are enabled."""
        cfg = _make_bioen_config()
        cfg.update({
            "output.bioen.ingest.enabled": "true",
            "output.bioen.maint.enabled": "true",
            "output.bioen.rho.enabled": "true",
            "output.bioen.sizeinf.enabled": "true",
        })
        _run_bioen_sim(cfg, tmp_path)

        bioen_dir = tmp_path / "Bioen"
        assert bioen_dir.exists(), "Bioen/ directory should be created"

        sp_names = ["Anchovy", "Sardine"]
        expected_labels = ["meanEnet", "ingestion", "maintenance", "rho", "sizeInf"]
        for label in expected_labels:
            for sp in sp_names:
                path = bioen_dir / f"osmose_{label}_{sp}_Simu0.csv"
                assert path.exists(), f"Expected {path.name} to exist"

    def test_disabled_flags_no_csv(self, tmp_path):
        """Disabled output flags do not produce CSVs for those outputs."""
        cfg = _make_bioen_config()
        # Leave all optional flags at default (False)
        _run_bioen_sim(cfg, tmp_path)

        bioen_dir = tmp_path / "Bioen"
        assert bioen_dir.exists(), "Bioen/ directory should still be created for meanEnet"

        sp_names = ["Anchovy", "Sardine"]
        # meanEnet is always written
        for sp in sp_names:
            assert (bioen_dir / f"osmose_meanEnet_{sp}_Simu0.csv").exists()

        # Optional outputs should NOT exist
        for label in ["ingestion", "maintenance", "rho", "sizeInf"]:
            for sp in sp_names:
                path = bioen_dir / f"osmose_{label}_{sp}_Simu0.csv"
                assert not path.exists(), f"{path.name} should not exist when flag is disabled"

    def test_no_bioen_no_dir(self, tmp_path):
        """Non-bioen config: Bioen/ directory is not created."""
        cfg = {
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "2",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "Anchovy",
            "species.linf.sp0": "15.0",
            "species.k.sp0": "0.4",
            "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
        }
        config = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        outputs = simulate(config, grid, rng)
        write_outputs(outputs, tmp_path, config, prefix="osmose")

        bioen_dir = tmp_path / "Bioen"
        assert not bioen_dir.exists(), "Bioen/ directory should not be created when bioen is disabled"

    def test_compute_energy_budget_returns_six(self):
        """compute_energy_budget now returns 6 values."""
        n = 10
        result = compute_energy_budget(
            ingestion=np.full(n, 1e-4),
            weight=np.full(n, 1e-5),
            gonad_weight=np.zeros(n),
            age_dt=np.full(n, 24, dtype=np.int32),
            length=np.full(n, 10.0),
            temp_c=np.full(n, 15.0),
            assimilation=0.7,
            c_m=0.001,
            beta=0.75,
            eta=1.4,
            r=0.45,
            m0=4.5,
            m1=1.8,
            e_maint_energy=0.63,
            phi_t=np.ones(n),
            f_o2=np.ones(n),
            n_dt_per_year=24,
            e_net_avg=np.zeros(n),
        )
        assert len(result) == 6, f"Expected 6 return values, got {len(result)}"
        dw, dg, e_net, e_gross, e_maint, rho = result
        assert dw.shape == (n,)
        assert dg.shape == (n,)
        assert e_net.shape == (n,)
        assert e_gross.shape == (n,)
        assert e_maint.shape == (n,)
        assert rho.shape == (n,)
        # e_gross should be positive (ingestion * assimilation * phi_T * f_O2)
        assert np.all(e_gross >= 0)
        # e_maint should be positive
        assert np.all(e_maint >= 0)
        # rho should be in [0, 1]
        assert np.all(rho >= 0) and np.all(rho <= 1)

    def test_step_output_has_new_fields(self):
        """StepOutput dataclass has all 4 new bioen fields."""
        cfg = _make_bioen_config()
        cfg.update({
            "output.bioen.ingest.enabled": "true",
            "output.bioen.maint.enabled": "true",
            "output.bioen.rho.enabled": "true",
            "output.bioen.sizeInf.enabled": "true",
            "simulation.time.nyear": "1",
        })
        config = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        outputs = simulate(config, grid, rng)

        for o in outputs:
            # All bioen arrays should be present and the right shape
            assert o.bioen_e_net_by_species is not None
            assert o.bioen_e_net_by_species.shape == (2,)
            assert o.bioen_ingestion_by_species is not None
            assert o.bioen_ingestion_by_species.shape == (2,)
            assert o.bioen_maint_by_species is not None
            assert o.bioen_maint_by_species.shape == (2,)
            assert o.bioen_rho_by_species is not None
            assert o.bioen_rho_by_species.shape == (2,)
            assert o.bioen_size_inf_by_species is not None
            assert o.bioen_size_inf_by_species.shape == (2,)

    def test_csv_content_is_numeric(self, tmp_path):
        """All bioen CSVs contain valid numeric data (Time column + value column)."""
        import pandas as pd

        cfg = _make_bioen_config()
        cfg.update({
            "output.bioen.ingest.enabled": "true",
            "output.bioen.maint.enabled": "true",
            "output.bioen.rho.enabled": "true",
            "output.bioen.sizeInf.enabled": "true",
            "simulation.time.nyear": "1",
        })
        _run_bioen_sim(cfg, tmp_path)

        bioen_dir = tmp_path / "Bioen"
        for csv_path in bioen_dir.glob("*.csv"):
            df = pd.read_csv(csv_path)
            assert "Time" in df.columns, f"{csv_path.name}: missing Time column"
            assert len(df.columns) == 2, f"{csv_path.name}: expected exactly 2 columns"
            assert df.shape[0] > 0, f"{csv_path.name}: no rows"
            # All values should be finite numbers
            assert df.select_dtypes(include="number").notna().all().all(), \
                f"{csv_path.name}: contains NaN values"
