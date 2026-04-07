"""Shared test utilities for OSMOSE UI tests."""

from typing import Any

_MISSING = object()


def make_fake_input(input_id: str, value: Any):
    """Create a FakeInput that only responds to a specific input ID."""

    class FakeInput:
        def __getattr__(self, name: str):
            if name == input_id:
                return lambda: value
            raise AttributeError(name)

    return FakeInput()


def make_catch_all_input(value: Any):
    """Create a FakeInput that returns the same value for any attribute."""

    class FakeInput:
        def __getattr__(self, name: str):
            return lambda: value

    return FakeInput()


def make_multi_input(default: Any = _MISSING, **kwargs: Any):
    """Create a FakeInput that returns different values per input ID.

    Usage:
        make_multi_input(foo=42, bar="hello")          # raises AttributeError for others
        make_multi_input(foo=42, default=None)          # returns None for others
        make_multi_input(foo=42, default=False)         # returns False for others
    """

    class FakeInput:
        def __getattr__(self, name: str):
            if name in kwargs:
                return lambda: kwargs[name]
            if default is not _MISSING:
                return lambda: default
            raise AttributeError(name)

    return FakeInput()
