"""ICES biomass target data model and CSV loader."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BiomassTarget:
    """A calibration target for one species, typically from ICES stock assessments."""

    species: str
    target: float
    lower: float
    upper: float
    weight: float = 1.0
    reference_point_type: str = "biomass"
    source: str = ""
    notes: str = ""


def load_targets(path: Path) -> tuple[list[BiomassTarget], dict]:
    """Load calibration targets from CSV.

    Skips ``#`` comment lines.  Lines starting with ``#!`` are parsed as
    key-value metadata (``str.split(":", 1)``).  The ``reference_point_type``,
    ``source``, and ``notes`` columns are optional for backward compatibility.

    Returns:
        (targets, metadata) tuple.
    """
    metadata: dict[str, str] = {}
    data_lines: list[str] = []

    text = path.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            payload = stripped[2:].strip()
            if ":" in payload:
                key, value = payload.split(":", 1)
                metadata[key.strip()] = value.strip()
        elif stripped.startswith("#"):
            continue
        elif stripped:
            data_lines.append(stripped)

    targets: list[BiomassTarget] = []
    if not data_lines:
        return targets, metadata

    reader = csv.DictReader(data_lines)
    for row in reader:
        targets.append(
            BiomassTarget(
                species=row["species"],
                target=float(row["target_tonnes"]),
                lower=float(row["lower_tonnes"]),
                upper=float(row["upper_tonnes"]),
                weight=float(row.get("weight", "1.0")),
                reference_point_type=row.get("reference_point_type", "biomass"),
                source=row.get("source", ""),
                notes=row.get("notes", ""),
            )
        )

    return targets, metadata
