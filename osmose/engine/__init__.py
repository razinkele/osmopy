"""Python OSMOSE simulation engine.

Provides a common Engine protocol for both Java (subprocess) and
Python (in-process vectorized) backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from osmose.runner import RunResult


@runtime_checkable
class Engine(Protocol):
    """Common interface for Java and Python OSMOSE engines."""

    def run(self, config: dict[str, str], output_dir: Path, seed: int = 0) -> RunResult: ...

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]: ...


class PythonEngine:
    """Vectorized in-process OSMOSE simulation engine."""

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend

    def run(self, config: dict[str, str], output_dir: Path, seed: int = 0) -> RunResult:
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.output import write_outputs
        from osmose.engine.simulate import simulate

        engine_config = EngineConfig.from_dict(config)

        # Try loading real grid from NetCDF, fall back to simple rectangular
        grid_file = config.get("grid.netcdf.file", "")
        mask_var = config.get("grid.var.mask", "mask")
        lat_var = config.get("grid.var.lat", "latitude")
        lon_var = config.get("grid.var.lon", "longitude")

        grid = None
        if grid_file:
            from pathlib import Path as P

            for base in [P("."), P("data/examples")]:
                path = base / grid_file
                if path.exists():
                    grid = Grid.from_netcdf(
                        path, mask_var=mask_var, lat_dim=lat_var, lon_dim=lon_var
                    )
                    break

        if grid is None:
            nx = int(config.get("grid.nlon", config.get("grid.ncol", "10")))
            ny = int(config.get("grid.nlat", config.get("grid.nrow", "10")))
            grid = Grid.from_dimensions(ny=ny, nx=nx)
        rng = np.random.default_rng(seed)

        outputs = simulate(engine_config, grid, rng)
        write_outputs(outputs, output_dir, engine_config)

        return RunResult(
            returncode=0,
            output_dir=output_dir,
            stdout="",
            stderr="",
        )

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]


class JavaEngine:
    """Wrapper around existing OsmoseRunner for Engine protocol compatibility."""

    def __init__(self, jar_path: Path, java_cmd: str = "java") -> None:
        self.jar_path = jar_path
        self.java_cmd = java_cmd

    def run(self, config: dict[str, str], output_dir: Path, seed: int = 0) -> RunResult:
        raise NotImplementedError(
            "JavaEngine.run() requires config file path — use OsmoseRunner directly for now"
        )

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]
