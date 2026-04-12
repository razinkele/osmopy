"""Shared pure helpers for UI page modules.

These are data-transformation functions extracted from reactive handlers
to enable direct testing. They must not import shiny or access reactive state.
"""

import re


def parse_nspecies(cfg: dict[str, str], default: int = 0) -> int:
    """Parse simulation.nspecies from a config dict, with fallback to default."""
    raw = cfg.get("simulation.nspecies", "") or ""
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return default


def count_map_entries(cfg: dict[str, str]) -> int:
    """Count non-null movement map entries in a config dict."""
    return sum(
        1
        for k, v in cfg.items()
        if re.match(r"movement\.file\.map\d+$", k)
        and isinstance(v, str)
        and v.strip()
        and v.strip().lower() not in ("null", "none")
    )
