"""Shared reactive application state for all UI pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from shiny import reactive
from shiny.types import SilentException

from osmose.logging import setup_logging
from osmose.runner import RunResult
from osmose.schema import build_registry
from ui.components.param_form import input_id_for_key


class ShinyInputs(Protocol):
    """Minimal interface for Shiny's runtime-generated ``input`` object."""

    def __getattr__(self, name: str) -> Any: ...


_log = setup_logging("osmose.state")

REGISTRY = build_registry()


class AppState:
    """Shared reactive state passed to all page server functions.

    Holds the current OSMOSE config, last run result, and output directory.
    All pages read/write through this single source of truth.
    """

    def __init__(self, scenarios_dir: Path = Path("data/scenarios")):
        self.config: reactive.Value[dict[str, str]] = reactive.Value({})
        self.output_dir: reactive.Value[Path | None] = reactive.Value(None)
        self.run_result: reactive.Value[RunResult | None] = reactive.Value(None)
        self.scenarios_dir: Path = scenarios_dir
        self.registry = REGISTRY
        self.jar_path: reactive.Value[str] = reactive.Value(
            "osmose-java/osmose_4.3.3-jar-with-dependencies.jar"
        )
        self.config_dir: reactive.Value[Path | None] = reactive.Value(None)
        self.loading: reactive.Value[bool] = reactive.Value(False)
        self.busy: reactive.Value[str | None] = reactive.Value(None)
        # C4 Phase B: cooperative cancellation token for the Python engine.
        # _run_python_engine constructs a fresh threading.Event per run and
        # publishes it here so handle_cancel() can flip it. None when no run
        # is in flight.
        import threading as _threading
        self.run_cancel_token: reactive.Value[_threading.Event | None] = reactive.Value(None)
        self.dirty: reactive.Value[bool] = reactive.Value(False)
        self.load_trigger: reactive.Value[int] = reactive.Value(0)
        self.config_name: reactive.Value[str] = reactive.Value("")
        self.species_names: reactive.Value[list[str]] = reactive.Value([])
        self.results_loaded: reactive.Value[bool] = reactive.Value(False)
        self.key_case_map: reactive.Value[dict[str, str]] = reactive.Value({})
        self.engine_mode: reactive.Value[str] = reactive.Value("java")

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
) -> dict[str, str]:
    """Read Shiny inputs for the given OSMOSE keys and update state.config.

    For each key, computes the input ID via key.replace(".", "_"), reads the
    value from input, and calls state.update_config() if non-None.

    Batches all updates into a single config.set() to avoid repeated
    reactive invalidations.

    Returns:
        Dict of keys that were actually updated with their new values.
    """
    # Check loading flag without creating a reactive dependency —
    # prevents sync effects from re-running when loading toggles,
    # which would overwrite config with stale input values before
    # ui.update_* messages reach the client.
    with reactive.isolate():
        if state.loading.get():
            _log.debug("sync_inputs: skipping — loading in progress")
            return {}
    changed: dict[str, str] = {}
    with reactive.isolate():
        cfg = dict(state.config.get())
    for key in keys:
        input_id = input_id_for_key(key)  # M13: canonical helper
        if not hasattr(input, input_id):
            continue
        try:
            val = getattr(input, input_id)()
        except TypeError:
            _log.warning("sync_inputs: TypeError reading input '%s'", input_id)
            continue
        if val is not None:
            new_val = str(val)
            # Preserve multi-value config entries (e.g. "2.3;1.8" for per-stage
            # arrays). The UI renders these as single-value inputs using the
            # field default, so writing back would clobber the original array.
            old_val = cfg.get(key, "")
            if ";" in old_val and ";" not in new_val:
                _log.debug(
                    "sync_inputs: preserving multi-value '%s' (old=%r, ui=%r)",
                    key,
                    old_val,
                    new_val,
                )
                continue
            changed[key] = new_val
    if changed:
        # Only set config if values actually differ
        actual_changes = {k: v for k, v in changed.items() if cfg.get(k) != v}
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
