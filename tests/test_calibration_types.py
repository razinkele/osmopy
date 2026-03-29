"""Tests for FreeParameter type safety."""

import enum

import pytest

from osmose.calibration.problem import FreeParameter, Transform


def test_transform_is_enum() -> None:
    """Transform should be an enum, not a bare string."""
    assert issubclass(Transform, enum.Enum)
    assert hasattr(Transform, "LINEAR")
    assert hasattr(Transform, "LOG")


def test_default_transform_is_linear() -> None:
    """FreeParameter defaults to LINEAR transform."""
    fp = FreeParameter(key="species.linf.sp0", lower_bound=10.0, upper_bound=100.0)
    assert fp.transform == Transform.LINEAR


def test_invalid_bounds_raises() -> None:
    """lower_bound >= upper_bound should raise ValueError."""
    with pytest.raises(ValueError, match="lower_bound.*must be less than.*upper_bound"):
        FreeParameter(key="species.linf.sp0", lower_bound=100.0, upper_bound=10.0)


def test_equal_bounds_raises() -> None:
    """lower_bound == upper_bound should raise ValueError."""
    with pytest.raises(ValueError, match="lower_bound.*must be less than.*upper_bound"):
        FreeParameter(key="species.linf.sp0", lower_bound=50.0, upper_bound=50.0)


def test_valid_log_transform() -> None:
    """Log transform with valid positive bounds should work."""
    fp = FreeParameter(
        key="species.linf.sp0",
        lower_bound=0.01,
        upper_bound=10.0,
        transform=Transform.LOG,
    )
    assert fp.transform == Transform.LOG
