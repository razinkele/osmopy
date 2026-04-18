"""Shared test utilities for OSMOSE tests."""

from pathlib import Path
from typing import Any

import numpy as np

from osmose.engine.state import SchoolState
from osmose.runner import OsmoseRunner

_MISSING = object()


# ---------------------------------------------------------------------------
# Engine test helpers
# ---------------------------------------------------------------------------


def _make_school(
    n: int = 1,
    sp: int = 0,
    abundance: float = 1000.0,
    length: float = 15.0,
    age_dt: int = 48,
    cell_x: int = 0,
    cell_y: int = 0,
) -> SchoolState:
    """Create a simple school state for testing."""
    state = SchoolState.create(n_schools=n, species_id=np.full(n, sp, dtype=np.int32))
    weight = 0.006 * length**3.0
    return state.replace(
        abundance=np.full(n, abundance),
        weight=np.full(n, weight),
        length=np.full(n, length),
        age_dt=np.full(n, age_dt, dtype=np.int32),
        cell_x=np.full(n, cell_x, dtype=np.int32),
        cell_y=np.full(n, cell_y, dtype=np.int32),
    )


class _ScriptRunner(OsmoseRunner):
    """OsmoseRunner that invokes Python scripts instead of Java.

    Used by multiple test modules to test runner behaviour without a real JVM.
    """

    def _build_cmd(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        **kwargs,
    ) -> list[str]:
        cmd = [self.java_cmd, str(self.jar_path), str(config_path)]
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if overrides:
            for key, value in overrides.items():
                cmd.append(f"-P{key}={value}")
        return cmd


# ---------------------------------------------------------------------------
# UI test helpers
# ---------------------------------------------------------------------------


def make_fake_input(input_id: str, value: Any):
    """Create a FakeInput that only responds to a specific input ID."""

    class FakeInput:
        def __getattr__(self, name: str):
            if name == input_id:
                return lambda: value
            raise AttributeError(name)

    return FakeInput()


def make_catch_all_input(value: Any):
    """Create a FakeInput that returns the same value for any attribute."""

    class FakeInput:
        def __getattr__(self, name: str):
            return lambda: value

    return FakeInput()


# ---------------------------------------------------------------------------
# Engine config test helpers
# ---------------------------------------------------------------------------

from dataclasses import replace  # noqa: E402

from osmose.engine.config import EngineConfig  # noqa: E402

# Minimum keys to satisfy EngineConfig.from_dict() for a 1-species, 0-background
# test config. Do NOT add grid.* keys: grid is constructed separately (see Task 3
# helper _make_grid_2x2_with_land) and is NOT parsed from the cfg dict by
# EngineConfig.from_dict.
_MINIMAL_CFG_DICT: dict[str, str] = {
    "simulation.nspecies": "1",
    "simulation.nbackground": "0",
    "simulation.time.ndtperyear": "24",
    "simulation.time.nyear": "1",
    # Species-0 minimum
    "species.name.sp0": "sp0",
    "species.lifespan.sp0": "10",
    # Von Bertalanffy growth
    "species.linf.sp0": "100.0",
    "species.k.sp0": "0.1",
    "species.t0.sp0": "0.0",
    "species.egg.size.sp0": "0.1",
    "species.length2weight.condition.factor.sp0": "0.01",
    "species.length2weight.allometric.power.sp0": "3.0",
    "species.vonbertalanffy.threshold.age.sp0": "0.0",
    # Predation
    "predation.ingestion.rate.max.sp0": "3.5",
    "predation.efficiency.critical.sp0": "0.57",
}


def make_minimal_engine_config(
    *,
    extra_cfg: dict[str, str] | None = None,
    **overrides,
) -> EngineConfig:
    """Build a small ``EngineConfig`` for unit tests with keyword overrides.

    ``extra_cfg`` injects/overrides raw config-dict keys before parsing
    (use this for any key the existing reader honors — e.g. ``output.X.enabled``).
    ``**overrides`` are applied via ``dataclasses.replace`` AFTER parsing,
    for fields not exposed via config keys.

    Raises AttributeError if an ``override`` names a non-existent
    ``EngineConfig`` field — this is intentional. Update the dataclass
    AND this helper together if you add a new field.
    """
    raw: dict[str, str] = dict(_MINIMAL_CFG_DICT)
    if extra_cfg:
        raw.update(extra_cfg)
    base = EngineConfig.from_dict(raw)
    if not overrides:
        return base
    # Validate overrides against declared fields
    declared = {f.name for f in base.__dataclass_fields__.values()}
    unknown = set(overrides) - declared
    if unknown:
        raise AttributeError(f"Unknown EngineConfig fields in overrides: {sorted(unknown)}")
    return replace(base, **overrides)


def make_multi_input(default: Any = _MISSING, **kwargs: Any):
    """Create a FakeInput that returns different values per input ID.

    Usage:
        make_multi_input(foo=42, bar="hello")          # raises AttributeError for others
        make_multi_input(foo=42, default=None)          # returns None for others
        make_multi_input(foo=42, default=False)         # returns False for others
    """

    class FakeInput:
        def __getattr__(self, name: str):
            if name in kwargs:
                return lambda: kwargs[name]
            if default is not _MISSING:
                return lambda: default
            raise AttributeError(name)

    return FakeInput()
