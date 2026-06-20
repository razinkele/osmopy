"""Pure, browser-free description of what each engine produces for a config.

Single source of truth for the Run-page capability panel. No Shiny imports;
the only engine dependency is ``java_engine_block_reason`` (already pure).
"""

from __future__ import annotations

from dataclasses import dataclass

from osmose.runner import java_engine_block_reason


@dataclass
class EngineCapability:
    engine: str  # "python" | "java"
    can_run: bool  # for THIS config
    block_reason: str | None  # why not, if can_run is False
    pages_populated: list[str]  # result/diagnostic pages that WILL have data
    pages_empty: list[str]  # pages that will NOT, for this engine+config
    notable_outputs: str  # one concise line of family-level differences


def _is_enabled(config: dict[str, str], key: str) -> bool:
    """True when a config flag reads as enabled (mirrors the engine convention)."""
    return str(config.get(key, "")).strip().lower() in ("true", "1")


# module flag -> page name, for Python conditional pages
_PYTHON_GATED_PAGES = [
    ("module.genetics.enabled", "Genetics"),
    ("module.bioeconomics.enabled", "Economic"),
    ("output.spatial.enabled", "Spatial Results"),
]

_PYTHON_NOTABLE = (
    "Not produced on the Python engine: sizeSpectrum, meanSize, meanTLByAge, "
    "yieldN, fishery-yield (run these on the Java engine)."
)
_JAVA_NOTABLE = (
    "Java run: no genetics, economics, or community size-spectrum outputs; "
    "cross-engine results are statistically equivalent, not bit-identical."
)


def _describe_python(config: dict[str, str]) -> EngineCapability:
    populated = ["Results", "Diagnostics"]
    empty: list[str] = []
    for flag, page in _PYTHON_GATED_PAGES:
        (populated if _is_enabled(config, flag) else empty).append(page)
    return EngineCapability(
        engine="python",
        can_run=True,
        block_reason=None,
        pages_populated=populated,
        pages_empty=empty,
        notable_outputs=_PYTHON_NOTABLE,
    )


def describe_engine(engine: str, config: dict[str, str]) -> EngineCapability:
    """Describe what ``engine`` will produce for ``config``. Total — never raises."""
    config = config or {}
    if engine == "python":
        return _describe_python(config)
    raise NotImplementedError  # Java handled in Task 3
