"""Shared pure helpers for UI page modules.

These are data-transformation functions extracted from reactive handlers
to enable direct testing. They must not import shiny or access reactive state.
"""


def parse_nspecies(cfg: dict[str, str], default: int = 0) -> int:
    """Parse simulation.nspecies from a config dict, with fallback to default."""
    raw = cfg.get("simulation.nspecies", "") or ""
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return default
