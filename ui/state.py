"""Shared reactive application state for all UI pages."""

from __future__ import annotations

from pathlib import Path

from shiny import reactive

from osmose.runner import RunResult
from osmose.schema import build_registry

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
        self.jar_path: reactive.Value[str] = reactive.Value("osmose-java/osmose.jar")
        self.loading: reactive.Value[bool] = reactive.Value(False)

    def update_config(self, key: str, value: str) -> None:
        """Update a single key in the config dict.

        Uses reactive.isolate to read current config without taking a
        reactive dependency — prevents infinite reactive loops when called
        from effects that also depend on inputs.
        """
        with reactive.isolate():
            cfg = dict(self.config.get())
        cfg[key] = value
        self.config.set(cfg)

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


def sync_inputs(
    input: object,
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
    if state.loading.get():
        return {}
    changed: dict[str, str] = {}
    for key in keys:
        input_id = key.replace(".", "_")
        try:
            val = getattr(input, input_id)()
        except (AttributeError, TypeError):
            continue
        if val is not None:
            changed[key] = str(val)
    if changed:
        with reactive.isolate():
            cfg = dict(state.config.get())
        cfg.update(changed)
        state.config.set(cfg)
    return changed
