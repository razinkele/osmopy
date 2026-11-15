"""Sync effects must never write inputs rendered for an earlier config load.

A tab that is hidden during a load does not re-render (Shiny suspends hidden outputs), so its
inputs keep the previous config's values. Its sync effect still fires (e.g. on n_species /
n_maps), and wrote those stale values into the new config: load EEC full, view Setup, load
Baltic -> 169 Baltic keys became EEC full's (2026-11-15, tests/test_e2e_config_reload.py).

Each render output marks the load generation it was built for (AppState.mark_rendered); each sync
writes only while its render is current. The inputs are always READ first: a reactive effect that
returns before reading them loses its dependencies and never syncs again.
"""

from __future__ import annotations

import ast
from pathlib import Path

from shiny import reactive

from ui.state import AppState, sync_inputs

_PAGES = Path(__file__).resolve().parent.parent / "ui" / "pages"


class _RecordingInput:
    """Fake Shiny input that answers every id and records which ids were read."""

    def __init__(self, value):
        self.value = value
        self.read: list[str] = []

    def __getattr__(self, name):
        if name in ("value", "read"):
            raise AttributeError(name)
        self.read.append(name)
        return lambda: self.value


def test_rendered_current_tracks_the_load_generation():
    state = AppState()
    with reactive.isolate():
        assert not state.rendered_current("p")  # never rendered
        state.mark_rendered("p")
        assert state.rendered_current("p")
        state.load_trigger.set(state.load_trigger.get() + 1)  # a load
        assert not state.rendered_current("p")
        state.mark_rendered("p")
        assert state.rendered_current("p")
        assert not state.rendered_current("p", "q")  # every name must be current


def test_sync_inputs_writes_nothing_from_a_stale_render_but_still_reads():
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "9"})
        state.mark_rendered("setup.simulation")
        state.load_trigger.set(state.load_trigger.get() + 1)  # new config loaded, tab hidden
        inp = _RecordingInput("14")  # the previous config's value, still in the input
        changed = sync_inputs(inp, state, ["simulation.nspecies"], rendered="setup.simulation")
        assert changed == {}
        assert state.config.get()["simulation.nspecies"] == "9"
        assert "simulation_nspecies" in inp.read  # read anyway: keeps the dependency


def test_sync_inputs_writes_once_the_render_is_current():
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "9"})
        state.mark_rendered("setup.simulation")
        changed = sync_inputs(
            _RecordingInput("10"), state, ["simulation.nspecies"], rendered="setup.simulation"
        )
        assert changed == {"simulation.nspecies": "10"}
        assert state.config.get()["simulation.nspecies"] == "10"


def test_sync_inputs_still_reads_while_busy():
    """Returning before the reads while busy would leave the effect with no dependencies, so the
    page's edits would never sync again after one busy period."""
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "9"})
        state.busy.set("running")
        inp = _RecordingInput("10")
        assert sync_inputs(inp, state, ["simulation.nspecies"]) == {}
        assert "simulation_nspecies" in inp.read
        assert state.config.get()["simulation.nspecies"] == "9"


def test_every_page_sync_is_guarded_by_a_marked_render():
    """Static guard: each sync_inputs call in ui/pages names its render (rendered=), and every
    render name a sync relies on is marked by some render function."""
    used: set[str] = set()
    marked: set[str] = set()
    unguarded: list[str] = []
    for path in sorted(_PAGES.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name == "sync_inputs":
                kw = {k.arg: k.value for k in node.keywords}
                if "rendered" not in kw:
                    unguarded.append(f"{path.name}:{node.lineno}")
                    continue
                value = kw["rendered"]
                for v in value.elts if isinstance(value, ast.Tuple) else [value]:
                    if isinstance(v, ast.Constant):
                        used.add(v.value)
            elif name == "rendered_current":
                used.update(a.value for a in node.args if isinstance(a, ast.Constant))
            elif name == "mark_rendered":
                marked.update(a.value for a in node.args if isinstance(a, ast.Constant))
    assert unguarded == [], f"sync_inputs without rendered=: {unguarded}"
    assert used, "no guarded syncs found: the test is not looking at the pages"
    assert used <= marked, f"render names used by a sync but never marked: {used - marked}"


def test_same_config_value_compares_by_value():
    from ui.state import same_config_value

    assert same_config_value("110.0", "110")  # numeric input returns an int when integral
    assert same_config_value("true", "True")  # switch returns a bool
    assert same_config_value("1e-3", "0.001")
    assert not same_config_value("0.5", "0.6")
    assert not same_config_value(None, "1")  # absent key: still written
    assert not same_config_value("2.3;1.8", "2.3")
    assert not same_config_value("cod", "Cod")  # only booleans ignore case


def test_viewing_a_tab_does_not_rewrite_a_value_in_another_format():
    """A sync that reads 110 from the numeric input must not rewrite the file's "110.0" or mark the
    config as changed (it did: every re-render normalised the loaded file's formatting)."""
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.time.nyear": "50.0"})
        state.dirty.set(False)
        state.mark_rendered("p")
        sync_inputs(_RecordingInput(50), state, ["simulation.time.nyear"], rendered="p")
        assert state.config.get()["simulation.time.nyear"] == "50.0"
        assert state.dirty.get() is False
