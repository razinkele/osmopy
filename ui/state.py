"""Shared reactive application state for all UI pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from htmltools import Tag
from shiny import reactive
from shiny import ui as shiny_ui
from shiny.types import SilentException

from osmose.config.multivalue import is_multi_value
from osmose.logging import setup_logging
from osmose.runner import RunResult
from osmose.schema import build_registry
from ui.components.param_form import (  # noqa: F401  (re-exported: pages and tests use them)
    input_id_for_key,
    same_config_value,
    shows_field_default,
    untouched_unset,
)


def render_gen_input_id(name: str) -> str:
    """Input id of the hidden input that carries render ``name``'s load generation (#283)."""
    return "osm_render_gen_" + name.replace(".", "_")


class ShinyInputs(Protocol):
    """Minimal interface for Shiny's runtime-generated ``input`` object."""

    def __getattr__(self, name: str) -> Any: ...


_log = setup_logging("osmose.state")

REGISTRY = build_registry()

# Anchored to the repo root rather than the process CWD so `shiny run` launched
# from any directory still finds the bundled scenarios.
_DEFAULT_SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"


class AppState:
    """Shared reactive state passed to all page server functions.

    Holds the current OSMOSE config, last run result, and output directory.
    All pages read/write through this single source of truth.
    """

    def __init__(self, scenarios_dir: Path = _DEFAULT_SCENARIOS_DIR):
        self.config: reactive.Value[dict[str, str]] = reactive.Value({})
        self.output_dir: reactive.Value[Path | None] = reactive.Value(None)
        self.run_result: reactive.Value[RunResult | None] = reactive.Value(None)
        self.scenarios_dir: Path = scenarios_dir
        self.registry = REGISTRY
        self.jar_path: reactive.Value[str] = reactive.Value(
            "osmose-java/osmose-4.4.1-jar-with-dependencies.jar"
        )
        self.config_dir: reactive.Value[Path | None] = reactive.Value(None)
        # M10 (2026-05-06): state.loading was consolidated into state.busy.
        # state.busy is the single source of truth — its value is None when
        # idle, or a non-empty string descriptor while a long-running task
        # blocks user input. Pages that previously called state.loading
        # now publish a meaningful descriptor via state.busy so the
        # status banner can show users *what* is blocking them.
        self.busy: reactive.Value[str | None] = reactive.Value(None)
        # C4 Phase B: cooperative cancellation token for the Python engine.
        # handle_run constructs a fresh threading.Event per run and publishes it
        # here so handle_cancel() can flip it. None when no run is in flight.
        import threading as _threading

        self.run_cancel_token: reactive.Value[_threading.Event | None] = reactive.Value(None)
        self.dirty: reactive.Value[bool] = reactive.Value(False)
        self.load_trigger: reactive.Value[int] = reactive.Value(0)
        self.config_name: reactive.Value[str] = reactive.Value("")
        self.species_names: reactive.Value[list[str]] = reactive.Value([])
        self.results_loaded: reactive.Value[bool] = reactive.Value(False)
        self.key_case_map: reactive.Value[dict[str, str]] = reactive.Value({})
        self.engine_mode: reactive.Value[str] = reactive.Value("python")
        # load_trigger generation each render output last built its inputs for (see
        # mark_rendered). A plain dict ON PURPOSE, not reactive: see mark_rendered.
        self._rendered_gen: dict[str, int] = {}

    def mark_rendered(self, name: str) -> Tag:
        """Record that render output ``name`` just built its inputs from the current config, and
        return the hidden input that carries this load generation. The render MUST include the
        returned tag in its UI.

        A sync effect writes that output's inputs only while ``rendered_current(name, input=...)``:
        a tab that is hidden during a load does not re-render (Shiny suspends hidden outputs), so
        its inputs still hold the PREVIOUS config's values, and writing them corrupted the new
        config (2026-11-15: load EEC full, view Setup, load Baltic -> 169 Baltic keys overwritten
        with EEC full's).

        The server-side mark alone is not enough (#283): it is set the moment the server builds the
        render, while the browser can still deliver values from the previous render afterwards. The
        Grid page, visible during every load, wrote a fresh session's placeholder corners over a
        loaded config that way. The browser binds a render's inputs together and sends their values
        in one batch with this generation input, so a value that arrives with the current
        generation belongs to the current render.

        Deliberately not reactive: if marking invalidated the sync, the sync would run right
        after the render, before the browser has sent the new inputs' values.
        """
        with reactive.isolate():
            gen = self.load_trigger.get()
            self._rendered_gen[name] = gen
        return shiny_ui.div(
            shiny_ui.input_text(render_gen_input_id(name), None, value=str(gen)),
            style="display:none",
        )

    def rendered_current(self, *names: str, input: ShinyInputs | None = None) -> bool:
        """True when every named render output was built for the current config load and, given
        ``input``, the browser reports that generation for each of them (see ``mark_rendered``).

        With ``input`` this READS the generation inputs, so a sync effect that calls it re-runs
        when a new render's values arrive. Call it before any early return, like the other reads.
        """
        client: list[object] = []
        if input is not None:
            for n in names:
                try:
                    client.append(getattr(input, render_gen_input_id(n))())
                except (AttributeError, SilentException):
                    client.append(None)
        with reactive.isolate():
            gen = self.load_trigger.get()
        if not all(self._rendered_gen.get(n) == gen for n in names):
            return False
        return input is None or all(str(c) == str(gen) for c in client)

    def get_config_value(self, key: str, default: str = "") -> str:
        """Read a single config value without copying the entire dict.

        Uses reactive.isolate() — no reactive dependency taken. Use in
        read-only render paths where only one key is needed.
        """
        with reactive.isolate():
            return self.config.get().get(key, default)

    def update_config(self, key: str, value: str) -> None:
        """Update a single key in the config dict.

        Uses reactive.isolate to read current config without taking a
        reactive dependency — prevents infinite reactive loops when called
        from effects that also depend on inputs.
        Only marks dirty when the value actually changes.
        """
        with reactive.isolate():
            cfg = dict(self.config.get())
        if cfg.get(key) == value:
            return
        cfg[key] = value
        self.config.set(cfg)
        self.dirty.set(True)

    def load_config(self, cfg: dict[str, str], case_map: dict[str, str] | None = None) -> list[str]:
        """Canonicalize a freshly-loaded config to 4.4.0 keys and set it as the active config.

        Returns the list of deprecated (old) keys seen, for one-time UI notification.
        """
        from osmose.config.aliases import canonicalize_config

        canon, deprecated = canonicalize_config(cfg)
        self.config.set(canon)
        if case_map is not None:
            self.key_case_map.set(dict(case_map))
        return deprecated

    def reset_to_defaults(self) -> None:
        """Reset config to default values from the schema registry.

        This replaces the entire config — any user edits are discarded.
        """
        nspecies_field = self.registry.get_field("simulation.nspecies")
        n_species = int(nspecies_field.default) if nspecies_field and nspecies_field.default else 3
        cfg: dict[str, str] = {}
        for field in self.registry.all_fields():
            if field.default is not None:
                if field.indexed:
                    for i in range(n_species):
                        key = field.resolve_key(i)
                        cfg[key] = str(field.default)
                else:
                    cfg[field.key_pattern] = str(field.default)
        self.config.set(cfg)
        self.dirty.set(False)


def sync_inputs(
    input: ShinyInputs,
    state: AppState,
    keys: list[str],
    rendered: str | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Read Shiny inputs for the given OSMOSE keys and update state.config.

    For each key, computes the input ID via key.replace(".", "_"), reads the
    value from input, and calls state.update_config() if non-None.

    Batches all updates into a single config.set() to avoid repeated
    reactive invalidations.

    ``rendered`` names the render output(s) that build these inputs (``state.mark_rendered``).
    While any of them was built for an earlier config load, nothing is written: the inputs hold
    the previous config's values.

    Viewing a page never writes: an input that still shows what the form renders for a key the
    config leaves unset (the schema default, or a placeholder) is untouched and skipped
    (``untouched_unset``). Defaults reach a config only by explicit seeding (``seed_new_species``
    when the species count goes up) or ``AppState.reset_to_defaults``. A real edit is written.

    Every input is READ before either gate (busy, stale render) is checked. A reactive effect
    re-runs only on what it read last time, so returning before the reads would drop its
    dependencies and stop the page's edits from ever syncing again.

    Returns:
        Dict of keys that were actually updated with their new values.
    """
    changed: dict[str, str] = {}
    with reactive.isolate():
        cfg = dict(state.config.get())
    for key in keys:
        input_id = input_id_for_key(key)  # M13: canonical helper
        if not hasattr(input, input_id):
            continue
        try:
            val = getattr(input, input_id)()
        except SilentException:
            # An input the page lists but never rendered (advanced/unrendered fields, keys for a
            # species added since the last render). Skip THIS key: letting it propagate aborted
            # the whole batch, so the Grid and Forcing global syncs never wrote anything.
            continue
        except TypeError:
            _log.warning("sync_inputs: TypeError reading input '%s'", input_id)
            continue
        if isinstance(val, (list, tuple, dict)):
            # A file-upload control (FILE_PATH / MATRIX fields render ui.input_file) returns a list
            # of upload records, not a path; str() of it would overwrite the configured file name.
            # Nothing stores schema-form uploads, so they are never synced.
            continue
        if val is not None:
            new_val = str(val)
            # Preserve multi-value config entries ("2.3;1.8", or "15, 5": the engine splits on
            # both). A single-value input cannot hold the array, so writing back would clobber it.
            old_val = cfg.get(key, "")
            field = REGISTRY.match_field(key)  # ',' separates arrays only on array fields
            if is_multi_value(old_val, field) and not is_multi_value(new_val, field):
                _log.debug(
                    "sync_inputs: preserving multi-value '%s' (old=%r, ui=%r)",
                    key,
                    old_val,
                    new_val,
                )
                continue
            if untouched_unset(field, key, cfg, new_val):
                continue
            changed[key] = new_val
    # Gate only the WRITE (the reads above keep this effect subscribed to its inputs).
    # M10: skip while a busy operation is in flight; isolated so toggling busy does not re-run
    # the effect and write input values before ui.update_* reaches the client.
    names = (rendered,) if isinstance(rendered, str) else rendered
    # Read the render generations with the other inputs, before either gate (see the docstring).
    current = state.rendered_current(*names, input=input) if names else True
    with reactive.isolate():
        if state.busy.get():
            _log.debug("sync_inputs: skipping — busy in progress")
            return {}
    if not current:
        _log.debug("sync_inputs: skipping — %s not yet current in the browser", names)
        return {}
    if changed:
        # Only set config if values actually differ (by value: see same_config_value)
        actual_changes = {
            k: v
            for k, v in changed.items()
            if not same_config_value(cfg.get(k), v, REGISTRY.match_field(k))
        }
        if actual_changes:
            cfg.update(actual_changes)
            state.config.set(cfg)
            state.dirty.set(True)
    return changed


def get_theme_mode(input: object) -> str:
    """Safely read theme_mode from Shiny input, defaulting to 'light'.

    Default matches the JS localStorage fallback in osmose.css toggleTheme().
    """
    try:
        mode = input.theme_mode()  # type: ignore[attr-defined]
        return mode if mode in ("dark", "light") else "light"
    except (SilentException, AttributeError, TypeError):
        # SilentException when input not initialized, AttributeError/TypeError otherwise
        return "light"
