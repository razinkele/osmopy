"""Pure, browser-free description of what each engine produces for a config, and of what each
engine offers the UI.

Single source of truth for the Run-page capability panel and for every "does this engine do X"
question a page asks. No Shiny imports; the engine dependencies are the two block-reason
functions (pure) and the C# engine's availability check, which loads nothing.

There are three engines, so "not Python" does not mean Java. A page asks by name:
:func:`is_java`, :func:`is_in_process`, :func:`engine_supports`.
"""

from __future__ import annotations

from dataclasses import dataclass

from osmose.runner import java_engine_block_reason

#: The values of ``AppState.engine_mode``, and how the UI names them.
ENGINE_MODES = ("python", "java", "csharp")
ENGINE_LABELS = {"python": "Python", "java": "Java", "csharp": "C#"}

#: What an engine offers the UI beyond a run, by feature name. A feature missing from an
#: engine's set is one its pages or panes must not offer.
_FEATURES: dict[str, frozenset[str]] = {
    # The live map draws the Python engine's school state at each step; the C# engine keeps its
    # schools to itself (it refuses a step observer), and Java runs in another process.
    "live_view": frozenset({"python"}),
    # Step-by-step progress comes from the same per-step observer.
    "step_progress": frozenset({"python"}),
    # The Diagnostics page reads the Python engine's timing record.
    "diagnostics": frozenset({"python"}),
    # Modules only the Python engine implements (the C# engine refuses such a config).
    "genetics": frozenset({"python"}),
    "economics": frozenset({"python"}),
    # Settings of the run panel.
    "thread_count": frozenset({"python", "csharp"}),
    "jar_settings": frozenset({"java"}),
}


def is_java(mode: str) -> bool:
    """The Java engine: a subprocess behind ``osmose.runner``, with a jar and Java options."""
    return mode == "java"


def is_in_process(mode: str) -> bool:
    """An engine that runs inside this process, on a worker thread: Python or C#."""
    return mode in ("python", "csharp")


def engine_supports(mode: str, feature: str) -> bool:
    """Whether the engine ``mode`` offers the UI ``feature``. An unknown feature is an error
    (a typo must not read as "unsupported"); an unknown engine supports nothing."""
    return mode in _FEATURES[feature]


def engine_label(mode: str) -> str:
    """The engine's name as the UI shows it."""
    return ENGINE_LABELS.get(mode, mode)


@dataclass
class EngineCapability:
    engine: str  # "python" | "java" | "csharp"
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
    "Not produced on the Python engine: sizeSpectrum, meanTLByAge, "
    "fishery-yield (run these on the Java engine)."
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


_JAVA_EMPTY_PAGES = ["Diagnostics", "Genetics", "Economic", "Spatial Results"]


def _describe_java(config: dict[str, str], jar_version: str | None = None) -> EngineCapability:
    block = java_engine_block_reason(config, jar_version)
    return EngineCapability(
        engine="java",
        can_run=block is None,
        block_reason=block,
        pages_populated=["Results"],
        pages_empty=list(_JAVA_EMPTY_PAGES),
        notable_outputs=_JAVA_NOTABLE,
    )


_CSHARP_EMPTY_PAGES = ["Diagnostics", "Genetics", "Economic", "Spatial Results"]
_CSHARP_NOTABLE = (
    "C# run: the Python engine's output families (life-stage, distributions, SSB, diet); no "
    "live view, step progress or diagnostics timing. It draws from its own random stream, so "
    "a run differs from the Python engine's run for run."
)


def _describe_csharp(config: dict[str, str]) -> EngineCapability:
    """The C# engine: unavailable (not built here), unable to run this config (a feature it does
    not implement, named), or able to."""
    block: str | None
    try:
        from osmose.engine_cs import csharp_engine_available
        from osmose.engine_cs.capabilities import csharp_engine_block_reason

        available, why = csharp_engine_available()
        if not available:
            block = f"The C# engine is not available on this machine: {why}"
        else:
            reason = csharp_engine_block_reason(config) if config else None
            block = (
                None if reason is None else f"The C# engine cannot run this configuration: {reason}"
            )
    except Exception as exc:  # total: a broken check must read as "cannot run", not crash the page
        block = f"The C# engine could not be checked: {exc}"
    return EngineCapability(
        engine="csharp",
        can_run=block is None,
        block_reason=block,
        pages_populated=["Results"],
        pages_empty=list(_CSHARP_EMPTY_PAGES),
        notable_outputs=_CSHARP_NOTABLE,
    )


def describe_engine(
    engine: str, config: dict[str, str], jar_version: str | None = None
) -> EngineCapability:
    """Describe what ``engine`` will produce for ``config``. Total — never raises.

    ``jar_version`` (the selected Java jar's version, e.g. "4.4.1") makes the Java capability
    version-aware: a >= 4.4.0 jar can run a staging-supported background config (C2).
    """
    config = config or {}
    if engine == "python":
        return _describe_python(config)
    if engine == "java":
        return _describe_java(config, jar_version)
    if engine == "csharp":
        return _describe_csharp(config)
    # Unknown engine — neutral, total fallback.
    return EngineCapability(
        engine=engine,
        can_run=False,
        block_reason=f"Unknown engine: {engine!r}",
        pages_populated=[],
        pages_empty=[],
        notable_outputs="",
    )
