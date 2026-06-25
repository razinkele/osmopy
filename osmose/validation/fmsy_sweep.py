"""Model-internal fishery reference points via a per-species yield-vs-F sweep."""

from __future__ import annotations

import pandas as pd

from osmose.engine.config import EngineConfig


class SharedFisheryError(RuntimeError):
    """Raised when a species' fishery lands >1 species (per-species sweep is ambiguous)."""


def _fisheries_enabled(cfg: dict) -> bool:
    return (
        cfg.get("module.multispecies.fisheries.enabled", "false").lower() == "true"
        and int(cfg.get("simulation.nfisheries", "0")) > 0
    )


def _species_to_fishery(cfg: dict) -> dict[str, int]:
    """Return species_name.lower() -> fishery column index (first catchability column > 0).

    Resolve the catchability file the same way the engine does: the reader injects
    the config dir under ``_osmose.config.dir``, and the engine resolves the relative
    path via :func:`osmose.engine.path_resolution.resolve_data_path`.
    """
    from osmose.engine.path_resolution import resolve_data_path

    catch_rel = cfg.get("fisheries.catchability.file")
    if not catch_rel:
        raise FileNotFoundError("fisheries.catchability.file not set")
    config_dir = cfg.get("_osmose.config.dir", "")
    catch_path = resolve_data_path(catch_rel, config_dir=config_dir)  # -> Path | None
    if catch_path is None:
        raise FileNotFoundError(f"catchability file not resolvable: {catch_rel!r}")
    df = pd.read_csv(catch_path, index_col=0)
    out: dict[str, int] = {}
    for r in range(len(df)):
        name = str(df.index[r]).strip().lower()
        for c in range(len(df.columns)):
            if float(df.iloc[r, c]) > 0:
                out[name] = c
                break
    return out


def fishing_override(
    base_config: dict, config: EngineConfig, species_idx: int
) -> tuple[str, float]:
    """Return ``(override_key, baseline_value)`` for the active fishing knob of species *i*.

    Parameters
    ----------
    base_config:
        Raw config dict as returned by :class:`~osmose.config.reader.OsmoseConfigReader`.
    config:
        Parsed :class:`~osmose.engine.config.EngineConfig` (built from the same dict).
    species_idx:
        Zero-based species index.

    Returns
    -------
    override_key:
        The config key that controls the fishing rate for this species:
        ``fisheries.rate.base.fsh{j}`` (v4 fisheries mode) or
        ``mortality.fishing.rate.sp{i}`` (legacy mode).
    baseline_value:
        The current value of ``config.fishing_rate[species_idx]``.

    Raises
    ------
    SharedFisheryError
        If the species' fishery is shared by more than one species (sweep is ambiguous).
    ValueError
        If the species is not found in the catchability matrix.
    """
    sp_name = config.species_names[species_idx].strip().lower()
    if _fisheries_enabled(base_config):
        s2f = _species_to_fishery(base_config)
        fsh = s2f.get(sp_name)
        if fsh is None:
            raise ValueError(f"species {sp_name!r} maps to no fishery")
        sharing = [n for n, j in s2f.items() if j == fsh]
        if len(sharing) > 1:
            raise SharedFisheryError(f"fishery {fsh} lands {len(sharing)} species: {sharing}")
        key = f"fisheries.rate.base.fsh{fsh}"
    else:
        key = f"mortality.fishing.rate.sp{species_idx}"
    return key, float(config.fishing_rate[species_idx])
