"""Tests for osmose.runner -- async OSMOSE Java engine runner.

Since we don't have the actual OSMOSE JAR for testing, we use a thin
subclass (_ScriptRunner) that overrides ``_build_cmd`` to invoke
Python scripts directly instead of going through ``java -jar``.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from osmose.runner import OsmoseRunner, RunResult


# ---------------------------------------------------------------------------
# Test helper: a runner subclass that calls Python scripts directly.
# In production the command is ``java [opts] -jar <jar> <config> [flags]``.
# For testing we substitute ``python <script> <config> [flags]`` so that
# the fake scripts can run without a real JVM.
# ---------------------------------------------------------------------------


class _ScriptRunner(OsmoseRunner):
    """OsmoseRunner variant that invokes scripts via ``python <script>``
    instead of ``java -jar <jar>`` so tests can use plain Python scripts."""

    def _build_cmd(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        **kwargs,
    ) -> list[str]:
        cmd = [self.java_cmd, str(self.jar_path), str(config_path)]
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if overrides:
            for key, value in overrides.items():
                cmd.append(f"-P{key}={value}")
        return cmd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_jar(tmp_path: Path) -> Path:
    """Create a fake 'JAR' that's actually a Python script mimicking OSMOSE output."""
    script = tmp_path / "fake_osmose.py"
    script.write_text(
        "import sys\n"
        'print("OSMOSE v4.3.3")\n'
        'print(f"Config: {sys.argv[1]}")\n'
        "for i in range(5):\n"
        '    print(f"Step {i+1}/5")\n'
        'print("Simulation complete")\n'
    )
    return script


@pytest.fixture
def failing_jar(tmp_path: Path) -> Path:
    """Create a fake JAR that exits with error."""
    script = tmp_path / "fail_osmose.py"
    script.write_text(
        'import sys\nprint("Error: config not found", file=sys.stderr)\nsys.exit(1)\n'
    )
    return script


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.csv"
    config.write_text("simulation.nspecies ; 1\n")
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_runner_successful_run(fake_jar: Path, fake_config: Path) -> None:
    runner = _ScriptRunner(jar_path=fake_jar, java_cmd=sys.executable)
    result = await runner.run(config_path=fake_config)
    assert result.returncode == 0
    assert "Simulation complete" in result.stdout
    assert isinstance(result, RunResult)


async def test_runner_captures_progress(fake_jar: Path, fake_config: Path) -> None:
    progress: list[str] = []
    runner = _ScriptRunner(jar_path=fake_jar, java_cmd=sys.executable)
    await runner.run(config_path=fake_config, on_progress=progress.append)
    assert any("Step" in line for line in progress)
    assert any("Simulation complete" in line for line in progress)


async def test_runner_handles_failure(failing_jar: Path, fake_config: Path) -> None:
    runner = _ScriptRunner(jar_path=failing_jar, java_cmd=sys.executable)
    result = await runner.run(config_path=fake_config)
    assert result.returncode != 0
    assert "Error" in result.stderr


async def test_runner_passes_overrides(tmp_path: Path) -> None:
    """Create a script that prints its arguments so we can verify overrides."""
    script = tmp_path / "args_osmose.py"
    script.write_text("import sys\nfor arg in sys.argv[1:]:\n    print(arg)\n")
    config = tmp_path / "config.csv"
    config.write_text("x ; 1\n")
    runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)
    result = await runner.run(
        config_path=config,
        overrides={"simulation.nspecies": "5"},
    )
    assert "-Psimulation.nspecies=5" in result.stdout


async def test_runner_passes_output_dir(fake_config: Path, tmp_path: Path) -> None:
    script = tmp_path / "args2.py"
    script.write_text("import sys\nfor arg in sys.argv[1:]:\n    print(arg)\n")
    runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)
    out = tmp_path / "myoutput"
    result = await runner.run(config_path=fake_config, output_dir=out)
    assert f"-Poutput.dir.path={out}" in result.stdout


async def test_runner_default_output_dir(fake_jar: Path, fake_config: Path) -> None:
    """When no output_dir is given, result should use config_path.parent / 'output'."""
    runner = _ScriptRunner(jar_path=fake_jar, java_cmd=sys.executable)
    result = await runner.run(config_path=fake_config)
    assert result.output_dir == fake_config.parent / "output"


async def test_runner_custom_output_dir_in_result(
    fake_jar: Path, fake_config: Path, tmp_path: Path
) -> None:
    """When output_dir is given, result should carry that path."""
    out = tmp_path / "custom_output"
    runner = _ScriptRunner(jar_path=fake_jar, java_cmd=sys.executable)
    result = await runner.run(config_path=fake_config, output_dir=out)
    assert result.output_dir == out


def test_get_java_version() -> None:
    """Just verify the method doesn't crash. Java may or may not be installed."""
    result = OsmoseRunner.get_java_version()
    # result is either a version string or None
    assert result is None or isinstance(result, str)


async def test_runner_cancel(tmp_path: Path) -> None:
    """Test that cancel terminates a long-running process."""
    script = tmp_path / "slow.py"
    script.write_text(
        "import time\n"
        'print("starting", flush=True)\n'
        "time.sleep(60)\n"  # Would run for 60 seconds
        'print("done")\n'
    )
    config = tmp_path / "config.csv"
    config.write_text("x ; 1\n")
    runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)

    async def run_and_cancel() -> RunResult:
        task = asyncio.create_task(runner.run(config_path=config))
        await asyncio.sleep(0.5)  # Let it start
        runner.cancel()
        return await task

    result = await asyncio.wait_for(run_and_cancel(), timeout=5.0)
    assert result.returncode != 0  # terminated


def test_build_cmd_includes_jar_flag() -> None:
    """Verify the production _build_cmd includes -jar in the right position."""
    runner = OsmoseRunner(jar_path=Path("/path/to/osmose.jar"), java_cmd="java")
    cmd = runner._build_cmd(
        config_path=Path("/data/config.csv"),
        output_dir=Path("/data/output"),
        java_opts=["-Xmx4g"],
        overrides={"simulation.nspecies": "5"},
    )
    assert cmd[0] == "java"
    assert cmd[1] == "-Xmx4g"
    assert cmd[2] == "-jar"
    assert cmd[3] == "/path/to/osmose.jar"
    assert cmd[4] == "/data/config.csv"
    assert "-Poutput.dir.path=/data/output" in cmd
    assert "-Psimulation.nspecies=5" in cmd


def test_build_cmd_minimal() -> None:
    """Verify _build_cmd with no optional arguments includes default -Xmx2g."""
    runner = OsmoseRunner(jar_path=Path("/path/to/osmose.jar"))
    cmd = runner._build_cmd(config_path=Path("/data/config.csv"))
    assert cmd == ["java", "-Xmx2g", "-jar", "/path/to/osmose.jar", "/data/config.csv"]


def test_get_java_version_not_found() -> None:
    """get_java_version returns None when the command doesn't exist."""
    result = OsmoseRunner.get_java_version(java_cmd="/nonexistent/java_binary_xyz")
    assert result is None


async def test_run_handles_none_stream(fake_config: Path, tmp_path: Path) -> None:
    """run() handles a process whose stdout is None without crashing."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_process = MagicMock()
    mock_process.stdout = None  # triggers the `if stream is None: return` guard
    mock_process.stderr = None
    mock_process.returncode = 0
    mock_process.wait = AsyncMock(return_value=0)

    runner = OsmoseRunner(jar_path=tmp_path / "fake.jar")

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        result = await runner.run(config_path=fake_config)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


# ---------------------------------------------------------------------------
# run_ensemble tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ensemble_jar(tmp_path: Path) -> Path:
    """Create a fake JAR that writes a marker file in the output directory.

    It parses -Poutput.dir.path=<dir> and -Psimulation.random.seed=<seed>
    from the command-line arguments, creates the output dir, and writes
    seed.txt inside it so the test can verify each replicate was invoked.
    """
    script = tmp_path / "ensemble_osmose.py"
    script.write_text(
        "import sys, os, pathlib\n"
        "output_dir = None\n"
        "seed = None\n"
        "for arg in sys.argv[1:]:\n"
        '    if arg.startswith("-Poutput.dir.path="):\n'
        '        output_dir = arg.split("=", 1)[1]\n'
        '    if arg.startswith("-Psimulation.random.seed="):\n'
        '        seed = arg.split("=", 1)[1]\n'
        "if output_dir:\n"
        "    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
        '    (pathlib.Path(output_dir) / "seed.txt").write_text(seed or "")\n'
        'print("done")\n'
    )
    return script


async def test_run_ensemble_creates_replicate_dirs(
    ensemble_jar: Path, fake_config: Path, tmp_path: Path
) -> None:
    runner = _ScriptRunner(jar_path=ensemble_jar, java_cmd=sys.executable)
    out = tmp_path / "ensemble_out"
    results = await runner.run_ensemble(config_path=fake_config, output_dir=out, n_replicates=3)
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.returncode == 0
        rep_dir = out / f"rep_{i}"
        assert rep_dir.exists()
        seed_file = rep_dir / "seed.txt"
        assert seed_file.exists()
        assert seed_file.read_text() == str(i)


async def test_run_ensemble_merges_overrides(
    ensemble_jar: Path, fake_config: Path, tmp_path: Path
) -> None:
    """Overrides should be passed through, with seed added."""
    # Use args-printing script to verify
    script = tmp_path / "args_ens.py"
    script.write_text(
        "import sys, pathlib\n"
        "output_dir = None\n"
        "for arg in sys.argv[1:]:\n"
        '    if arg.startswith("-Poutput.dir.path="):\n'
        '        output_dir = arg.split("=", 1)[1]\n'
        "if output_dir:\n"
        "    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
        "for arg in sys.argv[1:]:\n"
        "    print(arg)\n"
    )
    config = tmp_path / "config2.csv"
    config.write_text("x ; 1\n")
    runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)
    out = tmp_path / "ens_out2"
    results = await runner.run_ensemble(
        config_path=config,
        output_dir=out,
        n_replicates=2,
        overrides={"simulation.nspecies": "3"},
    )
    assert len(results) == 2
    # Check both custom override and seed are present
    assert "-Psimulation.nspecies=3" in results[0].stdout
    assert "-Psimulation.random.seed=0" in results[0].stdout
    assert "-Psimulation.random.seed=1" in results[1].stdout


async def test_run_ensemble_progress_callback(
    ensemble_jar: Path, fake_config: Path, tmp_path: Path
) -> None:
    """on_progress callback receives replicate index and line."""
    progress: list[tuple[int, str]] = []
    runner = _ScriptRunner(jar_path=ensemble_jar, java_cmd=sys.executable)
    out = tmp_path / "ens_progress"
    await runner.run_ensemble(
        config_path=fake_config,
        output_dir=out,
        n_replicates=2,
        on_progress=lambda i, line: progress.append((i, line)),
    )
    assert len(progress) > 0
    # Each entry should have the replicate index
    indices = {p[0] for p in progress}
    assert 0 in indices
    assert 1 in indices


async def test_run_ensemble_default_replicates(
    ensemble_jar: Path, fake_config: Path, tmp_path: Path
) -> None:
    """Default n_replicates=5."""
    runner = _ScriptRunner(jar_path=ensemble_jar, java_cmd=sys.executable)
    out = tmp_path / "ens_default"
    results = await runner.run_ensemble(config_path=fake_config, output_dir=out)
    assert len(results) == 5


async def test_run_timeout_kills_process(tmp_path: Path) -> None:
    """run() with timeout_sec kills a slow process and returns returncode=-1."""
    script = tmp_path / "slow_timeout.py"
    script.write_text('import time\nprint("starting", flush=True)\ntime.sleep(60)\nprint("done")\n')
    config = tmp_path / "config.csv"
    config.write_text("x ; 1\n")
    runner = _ScriptRunner(jar_path=script, java_cmd=sys.executable)
    result = await runner.run(config_path=config, timeout_sec=1)
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


def test_build_cmd_includes_verbose_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, verbose=True)
    assert "-verbose" in cmd


def test_build_cmd_includes_quiet_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, quiet=True)
    assert "-quiet" in cmd


def test_build_cmd_includes_xmx_default(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config)
    assert any(opt.startswith("-Xmx") for opt in cmd)


def test_build_cmd_xmx_override(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, java_opts=["-Xmx4g"])
    xmx_opts = [o for o in cmd if o.startswith("-Xmx")]
    assert len(xmx_opts) == 1
    assert xmx_opts[0] == "-Xmx4g"


def test_build_cmd_update_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, update=True)
    assert "-update" in cmd


def test_build_cmd_update_force_flag(tmp_path):
    jar = tmp_path / "osmose.jar"
    jar.touch()
    config = tmp_path / "config.csv"
    config.touch()
    runner = OsmoseRunner(jar_path=jar)
    cmd = runner._build_cmd(config, update=True, force=True)
    assert "-update" in cmd
    assert "-force" in cmd


def test_validate_java_opts_allows_safe_flags():
    from osmose.runner import validate_java_opts

    validate_java_opts(["-Xmx2g", "-Xms512m", "-Xss1m"])
    validate_java_opts(["-Dfoo.bar=baz"])
    validate_java_opts(["-XX:+UseG1GC"])
    validate_java_opts(["-server"])
    validate_java_opts(["-ea"])
    validate_java_opts([])


def test_validate_java_opts_rejects_unsafe_flags():
    from osmose.runner import validate_java_opts

    with pytest.raises(ValueError, match="[Uu]nsafe"):
        validate_java_opts(["-javaagent:/tmp/evil.jar"])

    with pytest.raises(ValueError, match="[Uu]nsafe"):
        validate_java_opts(["-agentlib:jdwp=transport=dt_socket"])
