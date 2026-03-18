"""Tests for the engine protocol and state foundation."""

from pathlib import Path

from osmose.engine import JavaEngine, PythonEngine


def test_python_engine_satisfies_protocol():
    """PythonEngine must have run() and run_ensemble() methods."""
    engine = PythonEngine()
    assert hasattr(engine, "run")
    assert hasattr(engine, "run_ensemble")


def test_java_engine_satisfies_protocol():
    """JavaEngine must have run() and run_ensemble() methods."""
    engine = JavaEngine(jar_path=Path("/fake.jar"))
    assert hasattr(engine, "run")
    assert hasattr(engine, "run_ensemble")
