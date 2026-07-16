"""Demo scenario generation and config version migration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from osmose.config.aliases import RENAMES_440
from osmose.logging import setup_logging

_log = setup_logging("osmose.demo")


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
    ("4.4.0", RENAMES_440),
]


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    if not v:
        return (0,)
    try:
        return tuple(int(x) for x in v.split("."))
    except (ValueError, AttributeError):
        _log.warning("Could not parse version %r; applying all migrations", v)
        return (0,)


def _data_root() -> Path:
    """Root of the bundled demo data.

    Overridable via ``OSMOSE_DATA_DIR`` — required for non-editable installs
    (wheel / Docker) where ``data/`` is NOT a sibling of the installed ``osmose``
    package, so the default ``Path(__file__).parent.parent / "data"`` misses it.
    """
    env = os.environ.get("OSMOSE_DATA_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "data"


def _bundled_data_dir(subdir: str) -> Path | None:
    """Return the bundled data subdir if present, else warn and return None.

    A missing data bundle previously fell through silently to a 5-line stub config
    (non-runnable) — surface it loudly so the cause is obvious.
    """
    d = _data_root() / subdir
    if d.exists():
        return d
    _log.warning(
        "Bundled demo data not found at %s — writing a minimal (non-runnable) stub config. "
        "Demos need the data bundle: use an editable install or set OSMOSE_DATA_DIR.",
        d,
    )
    return None


def list_demos() -> list[str]:
    """List available demo scenarios."""
    return [
        "baltic",
        "baltic_a2",
        "baltic_depensation",
        "bay_of_biscay",
        "eec",
        "eec_full",
        "minimal",
        "benguela",
    ]


# Per-model metadata for the UI model picker (title shown in the dropdown; the rest in the
# info modal). Keys MUST match list_demos(). The "engine" field is display text only; the actual
# Java/Python gating runs through osmose.runner.java_engine_block_reason (background-species
# staging support, depletable plankton, benguela forcing, ...).
DEMO_INFO: dict[str, dict[str, str]] = {
    "bay_of_biscay": {
        "title": "Bay of Biscay",
        "region": "NE Atlantic (Bay of Biscay)",
        "species": "8 focal species",
        "resources": "6 LTL/plankton groups",
        "engine": "Java + Python",
        "summary": "The OSMOSE reference example (anchovy, sardine, hake, …); runs on both engines.",
    },
    "eec": {
        "title": "Eastern English Channel",
        "region": "English Channel (reduced)",
        "species": "6 focal species",
        "resources": "no LTL resources",
        "engine": "Java + Python",
        "summary": "A reduced Eastern English Channel configuration; quick to run.",
    },
    "eec_full": {
        "title": "Eastern English Channel (full)",
        "region": "English Channel",
        "species": "14 focal species",
        "resources": "10 LTL + 1 background group",
        "engine": "Java 4.4.1 + Python",
        "summary": "The full 14-species EEC — the cross-engine parity benchmark. Runs on the "
        "Python engine and the Java 4.4.1 jar.",
    },
    "baltic": {
        "title": "Baltic Sea",
        "region": "Central/Eastern Baltic",
        "species": "8 focal species",
        "resources": "6 LTL + 2 background groups",
        "engine": "Java 4.4.1 + Python",
        "summary": "Cod, herring, sprat, flounder, perch, pike-perch, smelt, stickleback; uses "
        "background species + LTL forcing. Runs on the Python engine, and on the Java 4.4.1 jar "
        "(background staging).",
    },
    "baltic_a2": {
        "title": "Baltic Sea (A2-calibrated)",
        "region": "Central/Eastern Baltic",
        "species": "8 focal species",
        "resources": "6 LTL (depletable plankton) + 2 background groups",
        "engine": "Python",
        "summary": "The Baltic demo with depletable plankton (Chunk A2) and the converged DE "
        "mortality calibration. Best-achievable community fit, NOT fully ICES-calibrated: "
        "herring, sprat and stickleback land in-band and cod sits just above band, while the "
        "coastal percids (perch/pike-perch) stay structurally over at this grid resolution. A2 "
        "compresses the A2-off overshoot (17-400x) down to near-band. Python engine only "
        "(depletable plankton has no Java equivalent).",
    },
    "baltic_depensation": {
        "title": "Baltic Sea (depensation/Allee)",
        "region": "Central/Eastern Baltic",
        "species": "8 focal species",
        "resources": "6 LTL + 2 background groups",
        "engine": "Python",
        "summary": "The Baltic demo with the recruitment depensation/Allee gate enabled for cod "
        "— a low-SSB recruitment trap that can create bistability (a healthy and a collapsed cod "
        "state). Operating point (s50/theta/larval-M) is a placeholder pending the placement "
        "sweep. Python engine only (the depensation gate has no Java equivalent).",
    },
    "minimal": {
        "title": "Minimal",
        "region": "Toy configuration",
        "species": "2 focal species",
        "resources": "no LTL resources",
        "engine": "Java + Python",
        "summary": "A 2-species toy configuration for quick tests and smoke runs.",
    },
    "benguela": {
        "title": "Southern Benguela",
        "region": "SE Atlantic upwelling (Benguela)",
        "species": "10 focal species",
        "resources": "4 ROMS plankton groups",
        "engine": "Python",
        "summary": "Southern Benguela upwelling ecosystem (anchovy, sardine, redeye, hakes, "
        "snoek, …) forced by ROMS plankton; unfished. Python engine only. Uncalibrated example; "
        "the mesopelagic group declines over the 15-yr demo horizon.",
    },
}


def demo_info(name: str) -> dict[str, str] | None:
    """Return the metadata dict for a demo model, or None if unknown."""
    return DEMO_INFO.get(name)


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
        "baltic": _generate_baltic,
        "baltic_a2": _generate_baltic_a2,
        "baltic_depensation": _generate_baltic_depensation,
        "bay_of_biscay": _generate_bay_of_biscay,
        "eec": _generate_eec,
        "eec_full": _generate_eec_full,
        "minimal": _generate_minimal,
        "benguela": _generate_benguela,
    }
    gen = generators.get(scenario)
    if gen is None:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list_demos()}")
    return gen(output_dir)


def _generate_baltic(output_dir: Path) -> dict:
    """Generate Baltic Sea multi-species demo configuration."""
    data_dir = _bundled_data_dir("baltic")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        master = config_dir / "baltic_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 24\n"
            # Baltic is dynamically stable only over a ~15-yr horizon; past ~yr30 the config
            # collapses to herring+sprat on BOTH engines (calibration limit, not an engine bug).
            "simulation.time.nyear ; 15\n"
            "simulation.nspecies ; 8\n"
            "simulation.nresource ; 6\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "baltic_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def _generate_baltic_a2(output_dir: Path) -> dict:
    """Generate the A2-calibrated Baltic preset (depletable plankton + converged mortality).

    A thin overlay on the baltic demo: copy baltic's full config (grid/forcing/maps/sub-CSVs),
    then overlay the three baltic_a2 delta files (master + a2 mortality + a2 depletion). No
    NetCDFs are duplicated. Python-engine only (depletion has no Java equivalent).
    """
    data_dir = _bundled_data_dir("baltic")
    a2_dir = _bundled_data_dir("baltic_a2")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None and a2_dir is not None:
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
        shutil.copytree(a2_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "baltic_a2_all-parameters.csv").write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 15\n"
            "simulation.nspecies ; 8\n"
            "simulation.nresource ; 6\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "baltic_a2_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def _generate_baltic_depensation(output_dir: Path) -> dict:
    """Generate the depensation/Allee Baltic preset (SP1 overlay scaffold).

    A thin overlay on the baltic demo: copy baltic's full config (grid/forcing/maps/sub-CSVs),
    then overlay the baltic_depensation master (same includes as baltic + the depensation gate
    keys). No NetCDFs are duplicated. Python-engine only (the depensation gate has no Java
    equivalent). The operating point (s50/theta/larval-M) is a placeholder pending Task 8's
    placement sweep.
    """
    data_dir = _bundled_data_dir("baltic")
    dep_dir = _bundled_data_dir("baltic_depensation")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None and dep_dir is not None:
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
        shutil.copytree(dep_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "baltic_depensation_all-parameters.csv").write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 15\n"
            "simulation.nspecies ; 8\n"
            "simulation.nresource ; 6\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "baltic_depensation_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def _generate_bay_of_biscay(output_dir: Path) -> dict:
    """Generate Bay of Biscay 8-species demo."""
    # Copy from bundled examples if available
    examples_dir = _bundled_data_dir("examples")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if examples_dir is not None:
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
    data_dir = _bundled_data_dir("eec")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
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
    data_dir = _bundled_data_dir("eec_full")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
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
    data_dir = _bundled_data_dir("minimal")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
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


def _generate_benguela(output_dir: Path) -> dict:
    """Generate the Southern Benguela demo (Python-engine, unfished)."""
    data_dir = _bundled_data_dir("benguela")
    config_dir = output_dir / "config"
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)
    if data_dir is not None:
        shutil.copytree(data_dir, config_dir, dirs_exist_ok=True)
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "benguela_all-parameters.csv").write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.nspecies ; 10\n"
            "simulation.nresource ; 4\n"
            "simulation.ncpu ; 1\n"
        )
    return {"config_file": config_dir / "benguela_all-parameters.csv", "output_dir": sim_output}


def migrate_config(
    config: dict[str, str],
    target_version: str = "4.3.3",
) -> dict[str, str]:
    """Migrate config parameter names to a target OSMOSE version.

    Applies key renames sequentially from the config's current version
    through to target_version, following the Java engine's Releases.java chain.

    Note: migration chain currently covers up to v4.3.0; versions above have
    no key renames.
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
            keys_to_rename = [
                k for k in result if k == old_prefix or k.startswith(old_prefix + ".")
            ]
            for key in keys_to_rename:
                new_key = new_prefix + key[len(old_prefix) :]
                if new_key in result and new_key != key:
                    # Java updateKey: target already defined -> keep it, drop the old key.
                    result.pop(key)
                else:
                    result[new_key] = result.pop(key)

    result["osmose.version"] = target_version
    return result
