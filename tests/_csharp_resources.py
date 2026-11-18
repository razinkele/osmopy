"""The simulation's own ``ResourceState``, as the reference for the C# engine's resource tables.

``simulate()`` passes bottom-oxygen forcing to ``ResourceState`` (the oxygen-benthos K coupling).
A ``ResourceState`` built without it silently drops the coupling. Until task 2.5 the C# bundle
builder built it without oxygen, and a reference built the builder's way agreed with the builder
over exactly the coupling it lacked (found 2026-11-18, task 1.5). The builder now builds it as
``simulate()`` does; the rule stands for every reference: build it the way ``simulate()`` does,
through this helper, never the way the code under test does.
"""

from __future__ import annotations

from pathlib import Path

from osmose.engine.resources import ResourceState, oxygen_benthos_enabled
from osmose.engine.simulate import _load_oxygen_data


def simulation_resources(config, grid) -> ResourceState:
    """``ResourceState`` exactly as ``simulate()`` constructs it."""
    raw = config.raw_config
    config_dir = raw.get("_osmose.config.dir", "")
    oxygen = _load_oxygen_data(raw, Path(config_dir) if config_dir else None)
    return ResourceState(config=raw, grid=grid, oxygen=oxygen)


def oxygen_coupled(config, rs: ResourceState) -> int | None:
    """Index of the resource whose K bottom oxygen scales, or None when the coupling is off."""
    if not oxygen_benthos_enabled(config.raw_config):
        return None
    name = config.raw_config.get("ltl.oxygen.benthos.rsc", "Benthos")
    return next(i for i, s in enumerate(rs.species) if s.name == name)
