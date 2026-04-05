# osmose/calibration/configure.py
"""Auto-detect calibratable parameters from OSMOSE configuration."""

from __future__ import annotations

import re

# Calibratable patterns with default (lower, upper) bounds.
# Each pattern uses "spN" as a regex group for species index.
CALIBRATABLE_PATTERNS: dict[str, tuple[float, float]] = {
    r"mortality\.additional\.rate\.sp\d+": (0.001, 2.0),
    r"mortality\.additional\.larva\.rate\.sp\d+": (0.001, 10.0),
    r"mortality\.starvation\.rate\.max\.sp\d+": (0.001, 5.0),
    r"species\.k\.sp\d+": (0.01, 1.0),
    r"species\.linf\.sp\d+": (1.0, 300.0),
    r"predation\.ingestion\.rate\.max\.sp\d+": (0.5, 10.0),
    r"predation\.efficiency\.critical\.sp\d+": (0.1, 0.9),
    r"population\.seeding\.biomass\.sp\d+": (100, 1000000),
}


def configure_calibration(config: dict[str, str]) -> dict:
    """Scan config keys and return auto-detected calibratable parameters.

    Args:
        config: Dictionary of OSMOSE configuration key-value pairs.

    Returns:
        Dict with "params" key containing a list of parameter descriptors,
        each with keys: key, guess, lower, upper.
    """
    params: list[dict] = []

    for key, value in config.items():
        for pattern, (lower, upper) in CALIBRATABLE_PATTERNS.items():
            if re.fullmatch(pattern, key):
                try:
                    guess = float(value)
                except (ValueError, TypeError):
                    import logging

                    logging.getLogger(__name__).warning(
                        "Config value for %r is not numeric: %r, using midpoint %.3f",
                        key,
                        value,
                        (lower + upper) / 2,
                    )
                    guess = (lower + upper) / 2
                params.append(
                    {
                        "key": key,
                        "guess": guess,
                        "lower": lower,
                        "upper": upper,
                    }
                )
                break  # Each key matches at most one pattern

    return {"params": params}
