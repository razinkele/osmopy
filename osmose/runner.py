"""Manage OSMOSE Java engine execution."""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from osmose.logging import setup_logging

_log = setup_logging("osmose.runner")

_SAFE_JVM_PATTERNS = [
    re.compile(r"^-X(mx|ms|ss)\d+[kmgKMG]?$"),  # memory flags
    re.compile(r"^-D[\w.]+=[^;|&`$()]*$"),  # system properties
    re.compile(r"^-XX:[+-]?\w+(=[\w.]+)?$"),  # XX flags
    re.compile(r"^-server$"),  # JVM mode
    re.compile(r"^-client$"),
    re.compile(r"^-verbose:(gc|class|jni)$"),  # verbose modes
    re.compile(r"^-ea$"),  # enable assertions
    re.compile(r"^-da$"),  # disable assertions
]


def validate_java_opts(opts: list[str]) -> None:
    """Validate JVM options against a whitelist of safe patterns.

    Raises ValueError for any option that doesn't match a known-safe pattern.
    """
    for opt in opts:
        if not any(p.match(opt) for p in _SAFE_JVM_PATTERNS):
            raise ValueError(
                f"Unsafe JVM option: {opt!r}. Only memory, GC, and -D flags are allowed."
            )


@dataclass
class RunResult:
    """Result of an OSMOSE simulation run."""

    returncode: int
    output_dir: Path
    stdout: str
    stderr: str


class OsmoseRunner:
    """Execute the OSMOSE Java engine and stream progress."""

    def __init__(self, jar_path: Path, java_cmd: str = "java"):
        self.jar_path = jar_path
        self.java_cmd = java_cmd
        self._process: asyncio.subprocess.Process | None = None

    def _build_cmd(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        verbose: bool = False,
        quiet: bool = False,
        update: bool = False,
        force: bool = False,
    ) -> list[str]:
        """Build the command list for executing the OSMOSE engine."""
        opts = list(java_opts or [])
        if not any(o.startswith("-Xmx") for o in opts):
            opts.append("-Xmx2g")
        cmd = [self.java_cmd, *opts, "-jar", str(self.jar_path), str(config_path)]
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if verbose:
            cmd.append("-verbose")
        if quiet:
            cmd.append("-quiet")
        if update:
            cmd.append("-update")
        if force:
            cmd.append("-force")
        if overrides:
            for key, value in overrides.items():
                cmd.append(f"-P{key}={value}")
        return cmd

    async def run(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        on_progress: Callable[[str], None] | None = None,
        timeout_sec: int | None = None,
        verbose: bool = False,
        quiet: bool = False,
        update: bool = False,
        force: bool = False,
    ) -> RunResult:
        """Run the OSMOSE engine asynchronously.

        Args:
            config_path: Path to the master OSMOSE config file.
            output_dir: Override for output directory (passed as -P flag).
            java_opts: Extra JVM options (e.g., ["-Xmx4g"]).
            overrides: Extra parameter overrides (passed as -Pkey=value).
            on_progress: Callback for each line of stdout/stderr.
            timeout_sec: Maximum run time in seconds. None means no limit.
            verbose: Pass -verbose flag to the OSMOSE engine.
            quiet: Pass -quiet flag to the OSMOSE engine.
            update: Pass -update flag to trigger config version migration.
            force: Pass -force flag (used with update to overwrite).

        Returns:
            RunResult with returncode, stdout, stderr.
        """
        cmd = self._build_cmd(
            config_path,
            output_dir,
            java_opts,
            overrides,
            verbose=verbose,
            quiet=quiet,
            update=update,
            force=force,
        )
        _log.info("Starting OSMOSE: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def read_stream(
            stream: asyncio.StreamReader | None,
            lines_list: list[str],
        ) -> None:
            if stream is None:
                return
            async for line in stream:
                text = line.decode(errors="replace").rstrip()
                lines_list.append(text)
                if on_progress:
                    try:
                        on_progress(text)
                    except Exception:
                        _log.warning("Progress callback failed", exc_info=True)

        result_output_dir = output_dir or config_path.parent / "output"

        try:
            if timeout_sec is not None:
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(self._process.stdout, stdout_lines),
                        read_stream(self._process.stderr, stderr_lines),
                    ),
                    timeout=timeout_sec,
                )
            else:
                await asyncio.gather(
                    read_stream(self._process.stdout, stdout_lines),
                    read_stream(self._process.stderr, stderr_lines),
                )
        except asyncio.TimeoutError:
            _log.warning("OSMOSE run timed out after %ds, killing process", timeout_sec)
            try:
                self._process.kill()
            except ProcessLookupError:
                pass  # Process already exited
            await self._process.wait()
            return RunResult(
                returncode=-1,
                output_dir=result_output_dir,
                stdout="\n".join(stdout_lines),
                stderr=f"Run timed out after {timeout_sec}s (process killed)",
            )

        await self._process.wait()
        _log.info("OSMOSE finished with exit code %d", self._process.returncode)

        return RunResult(
            returncode=self._process.returncode if self._process.returncode is not None else -1,
            output_dir=result_output_dir,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )

    def cancel(self) -> None:
        """Terminate the running OSMOSE process."""
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass  # Process already exited between the returncode check and terminate()
            _log.info("OSMOSE run cancelled")

    async def migrate(
        self,
        config_path: Path,
        force: bool = False,
        timeout_sec: int | None = 120,
    ) -> RunResult:
        """Run the Java engine's built-in config version migration."""
        return await self.run(
            config_path,
            java_opts=[],
            update=True,
            force=force,
            timeout_sec=timeout_sec,
        )

    async def run_ensemble(
        self,
        config_path: Path,
        output_dir: Path,
        n_replicates: int = 5,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> list[RunResult]:
        """Run multiple replicates sequentially with different random seeds.

        Args:
            config_path: Path to the master OSMOSE config file.
            output_dir: Base output directory; replicates go into rep_0/, rep_1/, etc.
            n_replicates: Number of replicates to run.
            java_opts: Extra JVM options.
            overrides: Extra parameter overrides (seed is added automatically).
            on_progress: Callback ``(replicate_index, line)`` for each output line.

        Returns:
            List of RunResult, one per replicate.
        """
        results: list[RunResult] = []
        output_dir = Path(output_dir)

        for i in range(n_replicates):
            rep_dir = output_dir / f"rep_{i}"
            rep_overrides = dict(overrides) if overrides else {}
            rep_overrides["simulation.random.seed"] = str(i)

            _log.info("Starting replicate %d/%d", i + 1, n_replicates)

            progress_cb: Callable[[str], None] | None = None
            if on_progress:
                # Capture i by default-arg binding
                def _make_cb(idx: int) -> Callable[[str], None]:
                    return lambda line: on_progress(idx, line)

                progress_cb = _make_cb(i)

            result = await self.run(
                config_path=config_path,
                output_dir=rep_dir,
                java_opts=java_opts,
                overrides=rep_overrides,
                on_progress=progress_cb,
            )
            results.append(result)

        return results

    @staticmethod
    def get_java_version(java_cmd: str = "java") -> str | None:
        """Check if Java is installed and return version string."""
        try:
            result = subprocess.run(
                [java_cmd, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Java prints version to stderr
            return result.stderr.strip() or result.stdout.strip() or None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
