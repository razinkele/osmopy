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
