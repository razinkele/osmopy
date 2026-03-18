"""Python OSMOSE simulation engine.

Provides a common Engine protocol for both Java (subprocess) and
Python (in-process vectorized) backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from osmose.runner import RunResult


@runtime_checkable
class Engine(Protocol):
    """Common interface for Java and Python OSMOSE engines."""

    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult: ...

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]: ...


class PythonEngine:
    """Vectorized in-process OSMOSE simulation engine."""

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend

    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult:
        raise NotImplementedError("Phase 1 stub — simulation not yet implemented")

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]


class JavaEngine:
    """Wrapper around existing OsmoseRunner for Engine protocol compatibility."""

    def __init__(self, jar_path: Path, java_cmd: str = "java") -> None:
        self.jar_path = jar_path
        self.java_cmd = java_cmd

    def run(
        self, config: dict[str, str], output_dir: Path, seed: int = 0
    ) -> RunResult:
        raise NotImplementedError(
            "JavaEngine.run() requires config file path — use OsmoseRunner directly for now"
        )

    def run_ensemble(
        self,
        config: dict[str, str],
        output_dir: Path,
        n: int,
        base_seed: int = 0,
    ) -> list[RunResult]:
        return [self.run(config, output_dir, seed=base_seed + i) for i in range(n)]
