from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.scenarios import ScenarioManager

_NYEAR_KEY = "simulation.time.nyear"
_NDT_KEY = "simulation.time.ndtperyear"
_MOVE_RNG_KEY = "movement.randomseed.fixed"
_MORT_RNG_KEY = "stochastic.mortality.randomseed.fixed"
_DEFAULT_NYEAR = 10
_DEFAULT_NDT = 24


@dataclass(frozen=True)
class Basics:
    """Headline params the wizard sets on a source config.

    ``reproducible_rng`` controls the ``movement.randomseed.fixed`` and
    ``stochastic.mortality.randomseed.fixed`` engine flags together (a single
    toggle); a source with only one of the two flags set reads back as False,
    and applying writes both — mixed states are not represented.
    """

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


def validate_name(name: str, existing: set[str]) -> list[str]:
    """Problems with a proposed scenario name (empty list = valid)."""
    problems: list[str] = []
    n = (name or "").strip()
    if not n:
        return ["Name must not be empty"]
    if "/" in n or "\\" in n or ".." in n or n in (".", ".."):
        problems.append(f"Name contains invalid characters: {n!r}")
    if n in existing:
        problems.append(f"A scenario named '{n}' already exists")
    return problems


def default_description(kind: str, name: str, basics: Basics) -> str:
    src = f"{name} demo" if kind == "demo" else f"scenario '{name}'"
    return f"Created from {src}, {basics.nyear} yr"


def resolve_source(
    kind: str,
    name: str,
    *,
    scenarios_dir: Path,
    dest_dir: Path | None = None,
) -> ResolvedSource:
    """Resolve a wizard source to a config (+ dir + case_map + parent).

    demo: materialize into `dest_dir` (caller-owned, persistent) and read it.
    scenario: load the stored config dict (no files; config_dir is None).
    """
    if kind == "demo":
        if dest_dir is None:
            raise ValueError("dest_dir is required for a demo source")
        result = osmose_demo(name, dest_dir)
        config_file = Path(result["config_file"])
        reader = OsmoseConfigReader()
        cfg = reader.read(config_file)
        return ResolvedSource(
            kind="demo",
            name=name,
            config=cfg,
            config_dir=config_file.parent,
            case_map=dict(reader.key_case_map),
            parent=None,
        )
    if kind == "scenario":
        s = ScenarioManager(scenarios_dir).load(name)
        return ResolvedSource(
            kind="scenario",
            name=name,
            config=dict(s.config),
            config_dir=None,
            case_map=dict(s.key_case_map),
            parent=name,
        )
    raise ValueError(f"unknown source kind: {kind!r}")
