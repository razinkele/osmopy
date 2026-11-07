"""Stage an OSMOSE config for the Java engine.

Every Java run should go through :func:`stage_config_for_java`, so that each path writes the keys
the selected jar needs and none drifts from the others (GitHub #138 was the Run tab skipping a
step another path had). It lives in the library, not the UI, so callers without Shiny can use it:
calibration's Java path and the scripts. ``ui.pages.run`` re-exports these names.

This module must not import ``osmose.runner`` or ``ui``: ``osmose.java_background_staging``
documents the import cycle that would create.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from osmose.logging import setup_logging

_log = setup_logging("osmose.java_staging")


def _inject_random_movement_ncell(config: dict[str, str]) -> dict[str, str]:
    """Return a NEW dict with movement.distribution.ncell.spN injected.

    Works around a Java engine off-by-one bug in RandomDistribution.createRandomMap()
    where the engine accesses grid cell at index == grid size when ncell is not set.

    H3 (2026-05-06): refactored from in-place mutation to pure-return. The
    previous version mutated the caller's dict, which was surprising when
    invoked on a reactive value (the mutation could be observed mid-flight
    by other reactives). Callers now do `config = _inject_random_movement_ncell(config)`.
    """
    try:
        nlon = int(config.get("grid.nlon", "0"))
        nlat = int(config.get("grid.nlat", "0"))
    except ValueError:
        _log.warning(
            "Cannot inject random movement ncell: grid dimensions invalid (nlon=%r, nlat=%r)",
            config.get("grid.nlon"),
            config.get("grid.nlat"),
        )
        return dict(config)
    if nlon <= 0 or nlat <= 0:
        _log.warning(
            "Cannot inject random movement ncell: grid dimensions non-positive (nlon=%d, nlat=%d)",
            nlon,
            nlat,
        )
        return dict(config)
    total_cells = nlon * nlat
    out = dict(config)
    for key, value in config.items():
        if key.startswith("movement.distribution.method.sp") and value.strip() == "random":
            sp_suffix = key.split("movement.distribution.method.")[-1]
            ncell_key = f"movement.distribution.ncell.{sp_suffix}"
            if ncell_key not in out:
                out[ncell_key] = str(total_cells)
    return out


def _skip_entries_leading_to(target: Path):
    """``shutil.copytree`` ignore callback: skip the source entry that is, or contains, *target*.

    Without it, copying a directory into a stage that sits inside that directory never ends:
    copytree lists each directory when it reaches it, after the stage has been created, so the
    stage gets copied into itself until the path is too long.
    """
    target = target.resolve()

    def ignore(dirpath: str, names: list[str]) -> list[str]:
        here = Path(dirpath).resolve()
        return [n for n in names if here / n == target or here / n in target.parents]

    return ignore


def write_temp_config(
    config: dict[str, str],
    output_dir: Path,
    source_dir: Path | None = None,
    key_case_map: dict[str, str] | None = None,
    target_version: str = "4.3.3",  # bare default 4.3.3 (string-faithful); the run path passes
    # target_version_for_jar(jar_path) -> 4.4.1 (C1)
) -> Path:
    """Write config to a directory, copy data files, and return the master file path.

    If source_dir is provided, copies the entire directory tree first so that
    all ancillary files (NetCDF grids, movement maps, etc.) are available to
    the Java engine.  Then writes a single flat master config containing ALL
    parameters — without ``osmose.configuration.*`` sub-file references — so
    the Java engine reads only this one file and ignores any copied sub-configs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_dir and source_dir.is_dir():
        shutil.copytree(
            source_dir,
            output_dir,
            dirs_exist_ok=True,
            ignore=_skip_entries_leading_to(output_dir),  # a stage inside the source
        )

    # Work around Java off-by-one bug in RandomDistribution.createRandomMap():
    # when movement.distribution.ncell.spN is absent and method is "random",
    # the engine tries to access grid cell at index == grid size (out of bounds).
    # Auto-inject ncell = nlon * nlat - 1 for any species using random movement.
    # H3: now returns a new dict (was in-place mutation).
    config = _inject_random_movement_ncell(config)

    # Reverse-map canonical 4.4.0 keys to the target engine's spellings
    # (default 4.3.3, the bundled jar). The injected movement.* key is not
    # renamed, so it passes through untouched.
    from osmose.config.aliases import to_target_keys

    config = to_target_keys(config, target_version=target_version)

    # Write a single flat master file with all params, stripping sub-config
    # references to avoid the Java engine loading duplicate parameters from
    # both the master and the copied sub-config files.
    # Restore original key case so Java's case-sensitive parser works.
    case_map = key_case_map or {}
    master = output_dir / "osm_all-parameters.csv"
    lines = []
    for key, value in sorted(config.items()):
        if key.startswith(("osmose.configuration.", "_")):
            continue
        original_key = case_map.get(key, key)
        lines.append(f"{original_key} ; {value}\n")
    master.write_text("".join(lines))
    return master


def stage_config_for_java(
    config: dict[str, str],
    work_dir: Path,
    source_dir: Path | None = None,
    *,
    target_version: str = "4.3.3",
    key_case_map: dict[str, str] | None = None,
) -> tuple[Path, dict[str, str]]:
    """Stage *config* for the Java engine; return (master config path, -P overrides).

    Three ordered steps on the staged copy: write the flat master, stage background species,
    then reconcile names/matrices. Order matters — reconcile rewrites the master in place and
    must see the keys background staging adds, so it runs last.

    The reconcile pass is gated on >= 4.4.0 because it mirrors that engine's name stripping
    (``cod_west`` -> ``codwest``); applying it to the 4.3.3 jar would corrupt names it resolves
    verbatim. It is a no-op on a config that is already alphanumeric and matrix-consistent.

    Every Java staging path goes through here so they cannot drift apart again (the Run tab
    skipping the reconcile step was GitHub #138): the Run tab, calibration's Java path, and the
    scripts baltic_stability_certify.py, cross_engine_parity_440.py, validate_engines.py and
    baltic_440_smoke.py.

    Before 4.4.0, the master also gets the global ``species.biomass.nsteps.year`` that 4.3.3's
    resource forcing requires (``emit_433_resource_nsteps``). Without it the 4.3.3 jar exits at
    init on any config forcing resources from a NetCDF, the bundled Bay of Biscay one included.
    """
    from osmose.config.aliases import _numeric_version, emit_433_resource_nsteps

    is_44x = _numeric_version(target_version) >= _numeric_version("4.4.0")
    if not is_44x:
        config = emit_433_resource_nsteps(config)
    config_path = write_temp_config(
        config, work_dir, source_dir, key_case_map=key_case_map, target_version=target_version
    )
    overrides: dict[str, str] = {}

    try:
        n_background = int(float(config.get("simulation.nbackground", 0) or 0))
    except (TypeError, ValueError):
        n_background = 0
    if n_background > 0 and is_44x:
        from osmose.java_background_staging import stage_background_for_java

        overrides.update(stage_background_for_java(config_path.parent, config))

    if is_44x:
        from osmose.java_config_reconcile import reconcile_config_for_java

        reconcile_config_for_java(config_path.parent)

    return config_path, overrides
