"""Diagnostics for a failed Java run in ``scripts/validate_engines.py``.

The function under test replaced ``print(f"Java FAILED: {result.stderr[-500:]}")``,
which reliably hid the cause of every failure. OSMOSE logs its own fatal errors to
**stdout** (``osmose[severe] ...`` followed by a Java stack trace), while stderr
carries only the JVM's ``Picked up JAVA_TOOL_OPTIONS`` echo and SLF4J's binder
warning. On a proxied container that JAVA_TOOL_OPTIONS line alone exceeds 500
characters, so the old tail could not show anything else even in principle.

Measured 2026-09-27 against a locally built 4.4.1 jar: the real message was
"NETCDF_BIOMASS resource forcing is used but parameters are missing" and the
script printed none of it.
"""

from __future__ import annotations

import subprocess

from scripts.validate_engines import _java_failure_report

# The real stderr of a failed run on a proxied container: one very long JVM line
# plus SLF4J's three. Deliberately >500 chars so that a stderr-tail implementation
# cannot reach past it.
_JVM_NOISE = (
    "Picked up JAVA_TOOL_OPTIONS: -Djavax.net.ssl.trustStore=/root/.ccr/java-truststore.p12 "
    "-Dhttps.proxyHost=127.0.0.1 -Dhttps.proxyPort=40099 -Dhttp.nonProxyHosts=" + "127.*|" * 80
)
_SLF4J_NOISE = (
    'SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".\n'
    "SLF4J: Defaulting to no-operation (NOP) logger implementation\n"
    "SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details."
)
_SEVERE = "osmose[severe] NETCDF_BIOMASS resource forcing is used but parameters are missing"
_REAL_STDOUT = (
    "osmose[info] Software version: 4.4.1\n"
    "osmose[info] Simulation 0 started...\n"
    f"{_SEVERE}\n"
    "java.lang.Exception\n"
    "\tat fr.ird.osmose.resource.ResourceForcing.init(ResourceForcing.java:244)\n"
    "\tat fr.ird.osmose.Osmose.main(Osmose.java:492)"
)


def _result(stdout: str = "", stderr: str = "", returncode: int = 1):
    return subprocess.CompletedProcess(
        args=["java", "-jar", "osmose.jar"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_surfaces_severe_line_from_stdout() -> None:
    """The regression: fatal line on stdout, >500 chars of noise on stderr."""
    report = _java_failure_report(
        _result(stdout=_REAL_STDOUT, stderr=_JVM_NOISE + "\n" + _SLF4J_NOISE)
    )

    assert "NETCDF_BIOMASS resource forcing is used but parameters are missing" in report
    # And it is promoted, not merely present somewhere in a tail.
    assert "Engine reported:" in report


def test_old_stderr_tail_would_have_missed_it() -> None:
    """Pins down *why* the old implementation failed, so nobody reinstates it.

    Asserted against the same fixture the test above uses: the last 500 characters
    of stderr contain none of the diagnosis.
    """
    stderr = _JVM_NOISE + "\n" + _SLF4J_NOISE
    assert "NETCDF_BIOMASS" not in stderr[-500:]
    assert len(_JVM_NOISE) > 500


def test_filters_jvm_and_slf4j_noise() -> None:
    report = _java_failure_report(_result(stdout=_REAL_STDOUT, stderr=_JVM_NOISE))

    assert "Picked up JAVA_TOOL_OPTIONS" not in report
    assert "StaticLoggerBinder" not in report


def test_reads_stderr_too_when_that_is_where_the_error_is() -> None:
    """Not a stdout-only reader — a JVM crash writes to stderr instead."""
    report = _java_failure_report(
        _result(stdout="", stderr="Error: Could not find or load main class fr.ird.osmose.Osmose")
    )

    assert "Could not find or load main class" in report
    assert "stderr" in report


def test_includes_exit_code() -> None:
    assert "exit 137" in _java_failure_report(_result(stdout="boom", returncode=137))


def test_empty_streams_do_not_crash() -> None:
    """A killed JVM can yield nothing at all; the report must still be printable."""
    report = _java_failure_report(_result(stdout="", stderr=""))

    assert "Java FAILED" in report
    assert isinstance(report, str)


def test_none_streams_do_not_crash() -> None:
    """``capture_output=False`` leaves the streams None rather than ''."""
    report = _java_failure_report(_result(stdout=None, stderr=None))  # type: ignore[arg-type]

    assert "Java FAILED" in report


def test_tail_is_bounded() -> None:
    """A long run must not dump thousands of info lines into the terminal."""
    many = "\n".join(f"osmose[info] step {i}" for i in range(500))
    report = _java_failure_report(_result(stdout=many), tail=10)

    body = [ln for ln in report.splitlines() if "osmose[info] step" in ln]
    assert len(body) == 10
    assert "step 499" in report and "step 0" not in report


def test_multiple_severe_lines_are_capped() -> None:
    many_severe = "\n".join(f"osmose[severe] failure {i}" for i in range(20))
    report = _java_failure_report(_result(stdout=many_severe))

    promoted = [ln for ln in report.splitlines() if ln.strip().startswith("osmose[severe]")]
    # 5 in the promoted header, plus whatever the bounded tail repeats.
    assert len([ln for ln in promoted if "failure" in ln]) <= 5 + 25
    assert "failure 0" in report
