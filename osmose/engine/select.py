"""The in-process engines by name (C# engine spec 9.3).

``"python"`` is :class:`osmose.engine.PythonEngine`, ``"csharp"`` is
:class:`osmose.engine_cs.CSharpEngine`. Java is not here: it runs as a subprocess behind
:mod:`osmose.runner`. There is deliberately no environment variable or global default: every
result must come from an engine its caller named (spec 9.3, provenance).
"""

from __future__ import annotations

from typing import Any

IN_PROCESS_ENGINES = ("python", "csharp")


def get_engine(name: str, **kwargs: Any):
    """A new engine instance. ``kwargs`` go to its constructor (e.g. ``threads`` for C#)."""
    if name == "python":
        from osmose.engine import PythonEngine

        return PythonEngine(**kwargs)
    if name == "csharp":
        from osmose.engine_cs import CSharpEngine

        return CSharpEngine(**kwargs)
    raise ValueError(
        f"unknown in-process engine {name!r}: one of {', '.join(IN_PROCESS_ENGINES)} "
        "(Java runs through osmose.runner)"
    )
