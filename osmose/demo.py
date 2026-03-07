"""Demo scenario generation and config version migration."""

from __future__ import annotations

import shutil
from pathlib import Path


# Key renames between OSMOSE versions
_MIGRATIONS: dict[str, dict[str, str]] = {
    "4.3.0": {
        "simulation.nplankton": "simulation.nresource",
    },
}


def list_demos() -> list[str]:
    """List available demo scenarios."""
    return ["bay_of_biscay"]


def osmose_demo(scenario: str, output_dir: Path) -> dict:
    """Generate a demo OSMOSE configuration.

    Args:
        scenario: Demo name (e.g., "bay_of_biscay").
        output_dir: Directory to write demo files.

    Returns:
        Dict with keys: config_file, output_dir.
    """
    output_dir = Path(output_dir)

    if scenario == "bay_of_biscay":
        return _generate_bay_of_biscay(output_dir)
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list_demos()}")


def _generate_bay_of_biscay(output_dir: Path) -> dict:
    """Generate Bay of Biscay 3-species demo."""
    # Copy from bundled examples if available
    examples_dir = Path(__file__).parent.parent / "data" / "examples"
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    sim_output = output_dir / "output"
    sim_output.mkdir(parents=True, exist_ok=True)

    if examples_dir.exists():
        for f in examples_dir.glob("osm_*"):
            shutil.copy2(f, config_dir / f.name)
    else:
        # Generate minimal config
        master = config_dir / "osm_all-parameters.csv"
        master.write_text(
            "simulation.time.ndtperyear ; 24\n"
            "simulation.time.nyear ; 50\n"
            "simulation.nspecies ; 3\n"
            "simulation.nschool ; 20\n"
            "simulation.ncpu ; 1\n"
        )

    config_file = config_dir / "osm_all-parameters.csv"
    return {"config_file": config_file, "output_dir": sim_output}


def migrate_config(
    config: dict[str, str],
    target_version: str = "4.3.0",
) -> dict[str, str]:
    """Migrate config parameter names to a target OSMOSE version.

    Applies key renames for version compatibility.
    """
    current = config.get("osmose.version", "")
    if current == target_version:
        return dict(config)

    result = dict(config)
    renames = _MIGRATIONS.get(target_version, {})
    for old_key, new_key in renames.items():
        if old_key in result:
            result[new_key] = result.pop(old_key)

    result["osmose.version"] = target_version
    return result
