"""Every kwarg the Copernicus MCP server passes to ``copernicusmarine`` must exist on it.

Why this test exists: `mcp_servers/copernicus/server.py` called
``cm.subset(overwrite_output_data=True)`` long after copernicusmarine 2.x renamed that parameter
to ``overwrite``. ``cm.subset`` takes no ``**kwargs``, so the call raised ``TypeError`` on EVERY
invocation — and the server's ``except Exception`` turned that into an ordinary return string
("Download failed: ..."), so the tool reported a success-shaped response containing a failure.
Nothing caught it because `copernicusmarine` is not a declared dependency and the only other test
for this server skips whenever ``fastmcp`` is absent. See issue #162.

Two deliberate design choices:

* **AST, not import.** The server module imports ``fastmcp``, which is not in ``[dev]`` — importing
  it would make this test skip in exactly the environment that should be running it. Parsing the
  source needs neither ``fastmcp`` nor credentials nor a network.
* **Every ``cm.*`` call, not a hardcoded list.** The check reads whatever kwargs the source
  actually passes today, so it also covers call sites added after this test was written. A list
  of known-good names would go stale the same way the call site did.

Note that the drift was NOT uniform: ``cm.login`` still takes ``force_overwrite`` in 2.4.1, while
``subset``/``get`` moved to ``overwrite``. A blanket rename across the file would have broken the
two working ``login`` call sites, which is why this asserts per-function rather than per-name.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / "mcp_servers" / "copernicus" / "server.py"


def _cm_calls() -> list[tuple[str, list[str], int]]:
    """Every ``cm.<func>(...)`` call in the server: (func_name, kwarg_names, lineno)."""
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    out: list[tuple[str, list[str], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)):
            continue
        if fn.value.id != "cm":
            continue
        names = [kw.arg for kw in node.keywords if kw.arg is not None]
        out.append((fn.attr, names, node.lineno))
    return out


def test_server_calls_copernicusmarine_at_all() -> None:
    """Guard the guard: if the AST walk finds nothing, the real test below is vacuous.

    A rename of the ``cm`` import alias, or a refactor into a helper, would silently reduce this
    file to asserting nothing — the failure mode where a test stays green over code it no longer
    reaches.
    """
    calls = _cm_calls()
    assert calls, (
        f"no `cm.*(...)` calls found in {SERVER}; this test file has stopped checking anything"
    )


def test_every_kwarg_passed_to_copernicusmarine_exists() -> None:
    """No call may pass a kwarg copernicusmarine does not accept.

    This is the check that would have caught #162 at the version bump instead of at the next
    download attempt.
    """
    cm = pytest.importorskip(
        "copernicusmarine",
        reason=(
            "copernicusmarine is not installed. It is not a declared dependency (see issue #162) — "
            "install it in the env that actually runs the Copernicus MCP server, which is where a "
            "parameter rename breaks things. Run this test after upgrading copernicusmarine."
        ),
    )

    problems: list[str] = []
    for func_name, kwargs, lineno in _cm_calls():
        fn = getattr(cm, func_name, None)
        if fn is None:
            problems.append(f"server.py:{lineno}: copernicusmarine has no attribute {func_name!r}")
            continue
        params = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue  # **kwargs — the callee accepts anything, nothing to verify
        for name in kwargs:
            if name not in params:
                problems.append(
                    f"server.py:{lineno}: cm.{func_name}() does not accept {name!r} "
                    f"(available: {sorted(params)})"
                )

    assert not problems, "copernicusmarine API drift:\n  " + "\n  ".join(problems)
