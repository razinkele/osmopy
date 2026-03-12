"""Tooltip text for non-schema UI fields."""

# Manual tooltips for fields not in the schema registry.
# Keys match the input IDs or descriptive labels used in the UI.
MANUAL_TOOLTIPS: dict[str, str] = {
    "jar_path": (
        "Path to the OSMOSE Java JAR file. "
        "Place JAR files in the osmose-java/ directory."
    ),
    "java_opts": (
        "JVM options passed to the Java process. "
        "Common: -Xmx4g (max heap 4 GB), -Xms1g (initial heap 1 GB)."
    ),
    "run_timeout": (
        "Maximum time in seconds before the simulation is killed. "
        "Increase for large grids or many species."
    ),
    "param_overrides": (
        "Override config parameters for this run only. "
        "One key=value per line. Example: simulation.nyear=100"
    ),
    "output_dir": (
        "Directory containing OSMOSE output CSV files. "
        "Set automatically after a run, or enter a path manually."
    ),
    "n_species": (
        "Number of focal (modeled) species in the simulation. "
        "Each species gets its own growth, reproduction, and mortality parameters."
    ),
    "n_resources": (
        "Number of lower trophic level (plankton) resource groups. "
        "These are forced from external data, not dynamically modeled."
    ),
    "load_example": (
        "Select a bundled example configuration to load. "
        "This replaces the current configuration."
    ),
}
