from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_NYEAR_KEY = "simulation.time.nyear"
_NDT_KEY = "simulation.time.ndtperyear"
_MOVE_RNG_KEY = "movement.randomseed.fixed"
_MORT_RNG_KEY = "stochastic.mortality.randomseed.fixed"
_DEFAULT_NYEAR = 10
_DEFAULT_NDT = 24


@dataclass(frozen=True)
class Basics:
    nyear: int
    ndtperyear: int
    reproducible_rng: bool


@dataclass
class ResolvedSource:
    kind: str
    name: str
    config: dict[str, str]
    config_dir: Path | None
    case_map: dict[str, str]
    parent: str | None


def apply_basics(config: dict[str, str], basics: Basics) -> dict[str, str]:
    """Return a new config with the four headline keys set; everything else untouched."""
    out = dict(config)
    out[_NYEAR_KEY] = str(basics.nyear)
    out[_NDT_KEY] = str(basics.ndtperyear)
    flag = "true" if basics.reproducible_rng else "false"
    out[_MOVE_RNG_KEY] = flag
    out[_MORT_RNG_KEY] = flag
    return out


def _to_int(value: object, default: int) -> int:
    try:
        n = int(float(str(value)))
    except (ValueError, TypeError):
        return default
    return n if n >= 1 else default


def read_basics(config: dict[str, str]) -> Basics:
    """Parse the four headline keys from a config (sane fallbacks for missing/garbage)."""
    move = str(config.get(_MOVE_RNG_KEY, "false")).lower() == "true"
    mort = str(config.get(_MORT_RNG_KEY, "false")).lower() == "true"
    return Basics(
        nyear=_to_int(config.get(_NYEAR_KEY), _DEFAULT_NYEAR),
        ndtperyear=_to_int(config.get(_NDT_KEY), _DEFAULT_NDT),
        reproducible_rng=move and mort,
    )


def parse_source(value: str) -> tuple[str, str]:
    """Split a select value 'demo:<name>' / 'scenario:<name>' into (kind, name)."""
    for kind in ("demo", "scenario"):
        prefix = f"{kind}:"
        if value.startswith(prefix):
            return (kind, value[len(prefix) :])
    raise ValueError(f"unknown source value: {value!r}")


def source_choices(demos: list[str], scenarios: list[str]) -> dict[str, dict[str, str]]:
    """Grouped <optgroup> choices for input_select; omit the saved group when empty."""
    choices: dict[str, dict[str, str]] = {"Bundled demos": {f"demo:{d}": d for d in demos}}
    if scenarios:
        choices["Saved scenarios"] = {f"scenario:{s}": s for s in scenarios}
    return choices
