"""Python OSMOSE simulation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from osmose.results import OsmoseResults
from osmose.runner import RunResult


class PythonEngine:
    """Vectorized in-process OSMOSE simulation engine."""

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend

    def _resolve_grid(self, config: dict[str, str]):
        """Resolve grid from config — shared between run() and run_in_memory()."""
        from osmose.engine.grid import Grid

        grid_file = config.get("grid.netcdf.file", "")
        mask_var = config.get("grid.var.mask", "mask")
        lat_var = config.get("grid.var.lat", "latitude")
        lon_var = config.get("grid.var.lon", "longitude")

        if grid_file:
            config_dir = config.get("_osmose.config.dir", "")
            search_bases = [Path(".")]
            if config_dir:
                search_bases.insert(0, Path(config_dir))
            search_bases.append(Path("data/examples"))
            for base in search_bases:
                path = base / grid_file
                if path.exists():
                    return Grid.from_netcdf(
                        path, mask_var=mask_var, lat_dim=lat_var, lon_dim=lon_var
                    )
            searched = [str(b / grid_file) for b in search_bases]
            raise FileNotFoundError(
                f"Grid file '{grid_file}' not found in search paths: {searched}. "
                "Set grid.netcdf.file to an existing file or remove the key "
                "to use a rectangular grid."
            )

        nx = int(config.get("grid.nlon", config.get("grid.ncol", "10")))
        ny = int(config.get("grid.nlat", config.get("grid.nrow", "10")))
        return Grid.from_dimensions(ny=ny, nx=nx)

    def run(self, config: dict[str, str], output_dir: Path, seed: int = 0) -> RunResult:
        from osmose.engine.config import EngineConfig
        from osmose.engine.output import write_outputs
        from osmose.engine.rng import build_rng
        from osmose.engine.simulate import simulate

        engine_config = EngineConfig.from_dict(config)
        grid = self._resolve_grid(config)

        rng = np.random.default_rng(seed)
        movement_rngs = build_rng(seed, engine_config.n_species, engine_config.movement_seed_fixed)
        mortality_rngs = build_rng(
            seed + 1, engine_config.n_species, engine_config.mortality_seed_fixed
        )
        outputs = simulate(
            engine_config,
            grid,
            rng,
            movement_rngs=movement_rngs,
            mortality_rngs=mortality_rngs,
            output_dir=output_dir,
        )
        write_outputs(outputs, output_dir, engine_config, grid=grid)

        return RunResult(
            returncode=0,
            output_dir=output_dir,
            stdout="",
            stderr="",
        )

    def run_in_memory(
        self,
        config: dict[str, str],
        seed: int = 0,
    ) -> OsmoseResults:
        """Run the Python engine and return results as an in-memory OsmoseResults.

        Equivalent to run() except:
          - No output_dir argument.
          - No write_outputs() call.
          - Returns an OsmoseResults that serves DataFrames from the in-memory
            StepOutput list rather than from disk.

        Use this for calibration candidates, sensitivity analysis, or any other
        throughput-sensitive workflow where disk output is not needed.
        """
        from osmose.engine.config import EngineConfig
        from osmose.engine.rng import build_rng
        from osmose.engine.simulate import simulate

        engine_config = EngineConfig.from_dict(config)
        grid = self._resolve_grid(config)

        rng = np.random.default_rng(seed)
        movement_rngs = build_rng(seed, engine_config.n_species, engine_config.movement_seed_fixed)
        mortality_rngs = build_rng(
            seed + 1, engine_config.n_species, engine_config.mortality_seed_fixed
        )
        outputs = simulate(
            engine_config,
            grid,
            rng,
            movement_rngs=movement_rngs,
            mortality_rngs=mortality_rngs,
            output_dir=None,
        )
        return OsmoseResults.from_outputs(outputs, engine_config, grid)

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]
