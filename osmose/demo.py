"""Demo scenario generation and config version migration."""

from __future__ import annotations

import shutil
from pathlib import Path


# Migration chain: each entry is (introduced_in_version, {old_prefix: new_prefix})
# Renames are applied sequentially for configs older than each step version.
_MIGRATION_CHAIN: list[tuple[str, dict[str, str]]] = [
    (
        "3.2",
        {
            "population.initialization.biomass": "population.seeding.biomass",
            "population.initialization.abundance": "population.seeding.abundance",
        },
    ),
    (
        "3.3.3",
        {
            "grid.ncolumn": "grid.nlon",
            "grid.nline": "grid.nlat",
        },
    ),
    (
        "4.2.3",
        {
            "simulation.nplankton": "simulation.nresource",
            "plankton.name": "resource.name",
            "plankton.tl": "resource.tl",
            "plankton.size.min": "resource.size.min",
            "plankton.size.max": "resource.size.max",
            "plankton.accessibility2fish": "resource.accessibility2fish",
            "plankton.conversion2tons": "resource.conversion2tons",
            "plankton.file": "resource.file",
        },
    ),
    (
        "4.2.5",
        {
            "mortality.natural.rate": "mortality.additional.rate",
            "mortality.natural.larva.rate": "mortality.additional.larva.rate",
        },
    ),
    # No-op sentinel: marks the v4.3.0 version boundary without renaming any keys.
    (
        "4.3.0",
        {
            "simulation.restart.enabled": "simulation.restart.enabled",
        },
    ),
]


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    try:
        return tuple(int(x) for x in v.split("."))
    except (ValueError, AttributeError):
        return (0,)


def list_demos() -> list[str]:
    """List available demo scenarios."""
    return ["bay_of_biscay", "eec", "eec_full", "minimal"]


def osmose_demo(scenario: str, output_dir: Path) -> dict:
    """Generate a demo OSMOSE configuration.

    Args:
        scenario: Demo name (e.g., "bay_of_biscay", "eec", "minimal").
        output_dir: Directory to write demo files.

    Returns:
        Dict with keys: config_file, output_dir.
    """
    output_dir = Path(output_dir)

    generators = {
        "bay_of_biscay": _generate_bay_of_biscay,
        "eec": _generate_eec,
        "eec_full": _generate_eec_full,
        "minimal": _generate_minimal,
    }
    gen = generators.get(scenario)
    if gen is None:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list_demos()}")
    return gen(output_dir)


def _generate_bay_of_biscay(output_dir: Path) -> dict:
    """Generate Bay of Biscay 8-species demo."""
    # Copy from bundled examples if available
    examples_dir = Path(__file__).parent.parent / "data" / "examples"
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if examples_dir.exists():
        shutil.copytree(examples_dir, config_dir, dirs_exist_ok=True)
    else:
        # Generate minimal config
        config_dir.mkdir(parents=True, exist_ok=True)
        master = config_dir / "osm_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 50\n"
            "simulation.nspecies ; 8\n"
            "simulation.nschool ; 20\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "osm_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def _generate_eec(output_dir: Path) -> dict:
    """Generate Eastern English Channel 6-species demo."""
    data_dir = Path(__file__).parent.parent / "data" / "eec"
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir.exists():
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        master = config_dir / "osm_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 30\n"
            "simulation.nspecies ; 6\n"
            "simulation.nschool ; 20\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "osm_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def _generate_eec_full(output_dir: Path) -> dict:
    """Generate full EEC 14-species + 10 LTL research configuration.

    Based on GhassenH/OSMOSE_EEC — a calibrated Eastern English Channel model
    with 14 focal species, 10 plankton/benthos resource groups, 42 movement maps,
    and NetCDF LTL forcing.
    """
    data_dir = Path(__file__).parent.parent / "data" / "eec_full"
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir.exists():
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        master = config_dir / "eec_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 70\n"
            "simulation.nspecies ; 14\n"
            "simulation.nresource ; 10\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "eec_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def _generate_minimal(output_dir: Path) -> dict:
    """Generate minimal 2-species demo for testing and tutorials."""
    data_dir = Path(__file__).parent.parent / "data" / "minimal"
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir.exists():
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        master = config_dir / "osm_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 12\n"
            "simulation.time.nyear ; 10\n"
            "simulation.nspecies ; 2\n"
            "simulation.nschool ; 10\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "osm_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def migrate_config(
    config: dict[str, str],
    target_version: str = "4.3.3",
) -> dict[str, str]:
    """Migrate config parameter names to a target OSMOSE version.

    Applies key renames sequentially from the config's current version
    through to target_version, following the Java engine's Releases.java chain.
    """
    current = config.get("osmose.version", "")
    if current == target_version:
        return dict(config)

    current_tuple = _version_tuple(current)
    target_tuple = _version_tuple(target_version)

    result = dict(config)

    for step_version, renames in _MIGRATION_CHAIN:
        step_tuple = _version_tuple(step_version)
        if current and current_tuple >= step_tuple:
            continue
        if step_tuple > target_tuple:
            break
        for old_prefix, new_prefix in renames.items():
            if old_prefix == new_prefix:
                continue
            keys_to_rename = [k for k in result if k == old_prefix or k.startswith(old_prefix + ".")]
            for key in keys_to_rename:
                new_key = new_prefix + key[len(old_prefix):]
                result[new_key] = result.pop(key)

    result["osmose.version"] = target_version
    return result
