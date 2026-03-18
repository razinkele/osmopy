"""ResourceState: low trophic level (LTL) resource forcing.

Phase 1 stub — provides the interface that the simulation loop expects.
"""

from __future__ import annotations

from osmose.engine.grid import Grid


class ResourceState:
    """Container for LTL resource biomass per grid cell.

    Phase 1 is a no-op placeholder. Full implementation in Phase 4+.
    """

    def __init__(self, config: dict[str, str], grid: Grid) -> None:
        self.config = config
        self.grid = grid

    def update(self, step: int) -> None:
        """Load resource biomass for the given timestep. Phase 1 stub."""
