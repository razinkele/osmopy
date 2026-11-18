"""Plan task 2.7: the UI with three engines (C# engine spec 9.3).

With two engines the UI asked ``engine_mode != "python"`` and meant "Java" in some places and
"this needs the Python engine" in others. With a third engine those are different questions, so
every such test is an explicit predicate: ``is_java``, ``is_in_process``, ``engine_supports``.
These tests pin each predicate's answer for all three modes, that every page asks by predicate,
and that the C# engine runs on the in-process thread without the Python-only observer.

Nothing here loads the CLR: the engine is replaced by a stand-in.
"""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

import pytest

import osmose.engine_capabilities as caps
from osmose.engine import SimulationCancelled
from osmose.engine_capabilities import (
    ENGINE_LABELS,
    ENGINE_MODES,
    describe_engine,
    engine_label,
    engine_supports,
    is_in_process,
    is_java,
)

ROOT = Path(__file__).resolve().parents[1]
UI_SOURCES = [ROOT / "app.py", *sorted((ROOT / "ui").rglob("*.py"))]


# --------------------------------------------------------------------------- the predicates


def test_the_three_modes_and_their_labels():
    assert ENGINE_MODES == ("python", "java", "csharp")
    assert ENGINE_LABELS == {"python": "Python", "java": "Java", "csharp": "C#"}
    assert [engine_label(m) for m in ENGINE_MODES] == ["Python", "Java", "C#"]
    assert engine_label("fortran") == "fortran"  # an unknown mode is shown as it is


@pytest.mark.parametrize(
    ("mode", "java", "in_process"),
    [("python", False, True), ("java", True, False), ("csharp", False, True), ("fortran", False, False)],
)  # fmt: skip
def test_is_java_and_is_in_process(mode, java, in_process):
    assert is_java(mode) is java
    assert is_in_process(mode) is in_process


#: What each engine offers the UI: feature -> the engines that have it.
FEATURES = {
    "live_view": {"python"},
    "step_progress": {"python"},
    "diagnostics": {"python"},
    "genetics": {"python"},
    "economics": {"python"},
    "thread_count": {"python", "csharp"},
    "jar_settings": {"java"},
}


@pytest.mark.parametrize("feature", sorted(FEATURES))
@pytest.mark.parametrize("mode", ENGINE_MODES)
def test_engine_supports_every_feature_for_every_mode(feature, mode):
    assert engine_supports(mode, feature) is (mode in FEATURES[feature])


def test_engine_supports_knows_exactly_these_features_and_rejects_a_typo():
    assert set(caps._FEATURES) == set(FEATURES)
    assert engine_supports("fortran", "live_view") is False  # an unknown engine has nothing
    with pytest.raises(KeyError):
        engine_supports("python", "live_veiw")  # a typo must not read as "unsupported"


# --------------------------------------------------------------------- every page asks by name


def test_no_page_asks_whether_the_engine_is_not_python():
    """The rule of the task: with three engines, "not Python" is not a question."""
    pattern = re.compile(r"""engine_mode(\.get\(\))?\s*(!=|!==|==|===)\s*['"]python['"]""")
    hits = [
        f"{path.relative_to(ROOT)}:{i}: {line.strip()}"
        for path in UI_SOURCES
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not hits, hits


@pytest.mark.parametrize(
    ("page", "feature"),
    [("genetics", "genetics"), ("economic", "economics"), ("diagnostics", "diagnostics")],
)
@pytest.mark.parametrize("mode", ENGINE_MODES)
def test_the_python_only_pages_gate_on_their_feature(page, feature, mode):
    """Each of these pages shows its content for the Python engine alone, and asks by the
    feature's name; the C# engine, like Java, gets the hint to switch."""
    source = (ROOT / "ui" / "pages" / f"{page}.py").read_text(encoding="utf-8")
    assert f'if not engine_supports(state.engine_mode.get(), "{feature}"):' in source
    assert engine_supports(mode, feature) is (mode == "python")


def test_the_run_page_shows_each_engines_own_settings():
    text = (ROOT / "ui" / "pages" / "run.py").read_text(encoding="utf-8")
    # The jar settings for Java alone; the thread count and overrides for the in-process engines.
    assert text.count("\"input.engine_mode === 'java'\"") == 1
    assert text.count("\"input.engine_mode !== 'java'\"") == 1
    java, in_process = text.index("=== 'java'"), text.index("!== 'java'")
    assert java < text.index('"java_opts"') < in_process < text.index('"py_threads"')
    assert engine_supports("csharp", "thread_count") and not engine_supports("java", "thread_count")
    assert engine_supports("java", "jar_settings") and not engine_supports("csharp", "jar_settings")
    # The live view and the step progress are asked for by name.
    assert 'engine_supports(state.engine_mode.get(), "live_view")' in text
    # The live pane says so for an engine without a live view BEFORE it looks at the status an
    # earlier run left: "done" over an empty map would read as a live run.
    pane = text[text.index("def live_movement_status():") :]
    assert pane.index('engine_supports(state.engine_mode.get(), "live_view")') < pane.index(
        "_live_status_val.get()"
    )
    assert "Live view is available for the Python engine only." in pane[:900]
    assert 'engine_supports(engine_mode, "live_view")' in text
    assert 'engine_supports(engine_mode, "step_progress")' in text
    assert "if is_java(engine_mode):" in text and "if is_in_process(engine_mode):" in text
    assert 'elif engine_mode == "csharp":' in text  # the C# block reason, before a run starts


# ------------------------------------------------------------------- the header and the server


def test_the_server_accepts_the_csharp_mode_only_where_the_engine_is_installed():
    import app

    for offered in (True, False):
        assert app.accepts_engine_mode("python", offered) and app.accepts_engine_mode(
            "java", offered
        )
        assert app.accepts_engine_mode("csharp", offered) is offered
        for junk in ("fortran", "", None, 3, "PYTHON"):
            assert not app.accepts_engine_mode(junk, offered)
    # By default it follows what this installation has.
    assert app.accepts_engine_mode("csharp") is app.CSHARP_ENGINE_OFFERED


def test_the_header_offers_the_csharp_button_only_where_the_engine_is_installed():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "CSHARP_ENGINE_OFFERED = csharp_engine_available()[0]" in text
    button = text.index('id="engineBtnCsharp"')
    assert "if CSHARP_ENGINE_OFFERED" in text[button : button + 400]
    # The script knows the three buttons and falls back to Python for a mode without one (a
    # saved 'csharp' on a server without the engine).
    assert "csharp: document.getElementById('engineBtnCsharp')" in text
    assert "if (!buttons[mode]) {\n                mode = 'python';" in text
    # The Python-only pages stay gated on the Python engine, by name.
    assert (
        "if (mode === 'python') {\n                    el.classList.remove('osm-disabled');" in text
    )


# ------------------------------------------------------------------------ the capability panel


def test_the_csharp_capability_names_why_it_cannot_run(monkeypatch):
    import osmose.engine_cs.capabilities as cs_caps
    from osmose import engine_cs

    monkeypatch.setattr(engine_cs, "csharp_engine_available", lambda: (True, ""))
    monkeypatch.setattr(cs_caps, "csharp_engine_block_reason", lambda raw: None)
    ok = describe_engine("csharp", {"simulation.nspecies": "1"})
    assert ok.engine == "csharp" and ok.can_run and ok.block_reason is None
    assert ok.pages_populated == ["Results"]
    assert set(ok.pages_empty) == {"Diagnostics", "Genetics", "Economic", "Spatial Results"}
    assert "random stream" in ok.notable_outputs and "live view" in ok.notable_outputs

    monkeypatch.setattr(cs_caps, "csharp_engine_block_reason", lambda raw: "lmax_cap (phase 2)")
    blocked = describe_engine("csharp", {"simulation.nspecies": "1"})
    assert not blocked.can_run
    assert blocked.block_reason == "The C# engine cannot run this configuration: lmax_cap (phase 2)"

    monkeypatch.setattr(engine_cs, "csharp_engine_available", lambda: (False, "not built"))
    missing = describe_engine("csharp", {"simulation.nspecies": "1"})
    assert (
        not missing.can_run and "not available on this machine: not built" in missing.block_reason
    )


def test_the_csharp_capability_never_raises(monkeypatch):
    """``describe_engine`` is total: a check that blows up reads as "cannot run"."""
    from osmose import engine_cs

    def boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(engine_cs, "csharp_engine_available", boom)
    cap = describe_engine("csharp", {"a": "b"})
    assert not cap.can_run and "probe failed" in (cap.block_reason or "")
    monkeypatch.setattr(engine_cs, "csharp_engine_available", lambda: (True, ""))
    assert describe_engine("csharp", {}).can_run  # no config loaded yet: nothing to refuse


# ------------------------------------------------------------------- the in-process run thread


class _FakeEngine:
    def __init__(self, outcome="done"):
        self.outcome, self.calls = outcome, []

    def run(self, raw, output_dir, seed=0, **kwargs):
        self.calls.append({"raw": raw, "output_dir": output_dir, "seed": seed, **kwargs})
        if self.outcome == "cancel":
            raise SimulationCancelled("cancelled at step 3/24")
        if self.outcome == "refuse":
            raise RuntimeError("the C# engine cannot run this config: lmax_cap (phase 2)")
        return "RESULT"


def _run_thread(monkeypatch, outcome="done", n_threads=3):
    import ui.pages.run as run_mod

    fake, seen = _FakeEngine(outcome), {}

    def get_engine(name, **kwargs):
        seen.update(name=name, kwargs=kwargs)
        return fake

    monkeypatch.setattr(run_mod, "get_engine", get_engine)
    monkeypatch.setattr(
        run_mod, "apply_single_run_threads", lambda n: seen.setdefault("numba_threads", n)
    )
    done_q: queue.Queue = queue.Queue()
    token = threading.Event()
    run_mod._python_engine_thread({"k": "v"}, "out", token, None, done_q, n_threads, None, "csharp")
    return fake, seen, done_q.get_nowait(), token


def test_the_csharp_engine_runs_on_the_in_process_thread_without_an_observer(monkeypatch):
    fake, seen, outcome, token = _run_thread(monkeypatch)
    assert outcome == ("done", "RESULT", "")
    assert seen["name"] == "csharp" and seen["kwargs"] == {"threads": 3}
    assert "numba_threads" not in seen  # Numba's thread count is the Python engine's concern
    (call,) = fake.calls
    assert call == {"raw": {"k": "v"}, "output_dir": "out", "seed": 0, "cancel_token": token}
    assert "step_observer" not in call  # the C# engine refuses one


def test_auto_threads_resolve_to_a_real_count_for_the_csharp_engine(monkeypatch):
    from osmose.engine.thread_policy import physical_budget

    _, seen, _, _ = _run_thread(monkeypatch, n_threads=0)
    assert seen["kwargs"] == {"threads": physical_budget()} and physical_budget() >= 1


def test_a_cancel_and_a_refusal_are_reported_as_such(monkeypatch):
    _, _, outcome, _ = _run_thread(monkeypatch, "cancel")
    assert outcome == ("cancelled", None, "cancelled at step 3/24")
    _, _, outcome, _ = _run_thread(monkeypatch, "refuse")
    assert outcome == ("failed", None, "the C# engine cannot run this config: lmax_cap (phase 2)")


def test_the_python_engine_still_takes_the_observer_and_numbas_thread_count(monkeypatch):
    import ui.pages.run as run_mod

    seen: dict = {}

    class FakePython:
        def run(self, raw, output_dir, seed=0, cancel_token=None, step_observer=None):
            seen.update(observer=step_observer, token=cancel_token)
            return "PY"

    monkeypatch.setattr(run_mod, "PythonEngine", FakePython)
    monkeypatch.setattr(run_mod, "apply_single_run_threads", lambda n: seen.setdefault("numba", n))
    monkeypatch.setattr(run_mod, "get_engine", lambda *a, **k: pytest.fail("not the registry"))
    done_q: queue.Queue = queue.Queue()
    observer = object()
    run_mod._python_engine_thread({"k": "v"}, "out", None, observer, done_q, 5)
    assert done_q.get_nowait() == ("done", "PY", "")
    assert seen == {"numba": 5, "observer": observer, "token": None}
