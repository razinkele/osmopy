"""Stage-indexed accessibility matrix for predation.

Parses the OSMOSE accessibility CSV where row/column labels encode
species name and optional age thresholds (e.g., "cod < 0.4").
Provides fast lookup of matrix indices for each school based on
species identity and age.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class StageInfo:
    """Threshold and matrix index for one stage of a species."""

    threshold: float  # age threshold (years); inf for the final/only stage
    matrix_index: int  # row or column index in the raw matrix


@dataclass
class AccessibilityMatrix:
    """Stage-indexed predation accessibility matrix.

    The raw CSV has row labels (prey) and column labels (predator)
    that encode species names and age thresholds, e.g.:
        "lesserSpottedDogfish < 0.45" — schools younger than 0.45 years
        "lesserSpottedDogfish" — schools at or above the threshold

    For a species with label "X < T", stages are:
        stage 0: age < T  → matrix index of "X < T"
        stage 1: age >= T → matrix index of "X"

    For a species with no threshold (single label "X"):
        stage 0: all ages → matrix index of "X"
    """

    raw_matrix: NDArray[np.float64]  # shape (n_prey_stages, n_pred_stages)
    prey_lookup: dict[str, list[StageInfo]] = field(default_factory=dict)
    pred_lookup: dict[str, list[StageInfo]] = field(default_factory=dict)
    # Species name → index mapping (normalised names)
    _species_name_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, csv_path: str | Path, species_names: list[str]) -> AccessibilityMatrix:
        """Load and parse the accessibility CSV.

        Parameters
        ----------
        csv_path:
            Path to the semicolon-separated accessibility CSV.
        species_names:
            Ordered list of ALL species names (focal + background + resource)
            as they appear in the engine config.
        """
        df = pd.read_csv(csv_path, sep=";", index_col=0)
        raw_matrix = df.values.astype(np.float64)

        # Parse row labels (prey)
        prey_labels = [str(lbl).strip() for lbl in df.index]
        prey_lookup = _parse_labels(prey_labels)

        # Parse column labels (predator)
        pred_labels = [str(lbl).strip() for lbl in df.columns]
        pred_lookup = _parse_labels(pred_labels)

        # Build normalised name map for fuzzy matching
        name_map: dict[str, str] = {}
        all_label_names = set()
        for labels in (prey_labels, pred_labels):
            for lbl in labels:
                name, _ = _parse_label(lbl)
                all_label_names.add(name)

        for csv_name in all_label_names:
            norm = csv_name.strip().lower()
            name_map[norm] = csv_name

        return cls(
            raw_matrix=raw_matrix,
            prey_lookup=prey_lookup,
            pred_lookup=pred_lookup,
            _species_name_map=name_map,
        )

    def resolve_name(self, species_name: str) -> str | None:
        """Resolve a config species name to its CSV label name."""
        norm = species_name.strip().lower()
        return self._species_name_map.get(norm)

    def get_index(
        self,
        species_name: str,
        age_years: float,
        role: str = "prey",
    ) -> int:
        """Get the matrix row/column index for a species at a given age.

        Parameters
        ----------
        species_name:
            Species name as it appears in the CSV labels.
        age_years:
            School age in years.
        role:
            "prey" for row index, "pred" for column index.

        Returns
        -------
        Matrix index, or -1 if species not found.
        """
        lookup = self.prey_lookup if role == "prey" else self.pred_lookup
        stages = lookup.get(species_name)
        if stages is None:
            return -1
        # Stages are sorted by threshold ascending.
        # Return the last stage whose threshold is > age (i.e., age < threshold).
        # If age >= all thresholds, return the last stage (the "adult" / no-threshold one).
        for stage in stages:
            if age_years < stage.threshold:
                return stage.matrix_index
        # age >= all thresholds → return the last stage
        return stages[-1].matrix_index

    def compute_school_indices(
        self,
        species_id: NDArray[np.int32],
        age_dt: NDArray[np.int32],
        n_dt_per_year: int,
        all_species_names: list[str],
        role: str = "prey",
    ) -> NDArray[np.int32]:
        """Compute matrix row/col index for every school.

        Parameters
        ----------
        species_id:
            Species index per school.
        age_dt:
            Age in timesteps per school.
        n_dt_per_year:
            Timesteps per year (for age conversion).
        all_species_names:
            Full species name list (focal + background).
        role:
            "prey" or "pred".

        Returns
        -------
        Array of matrix indices, shape (n_schools,). -1 if not found.
        """
        n = len(species_id)
        indices = np.full(n, -1, dtype=np.int32)

        # Build per-species resolved name cache
        resolved: dict[int, str | None] = {}
        for sp_idx in range(len(all_species_names)):
            name = all_species_names[sp_idx]
            csv_name = self.resolve_name(name)
            resolved[sp_idx] = csv_name

        for i in range(n):
            sp = species_id[i]
            csv_name = resolved.get(sp)
            if csv_name is None:
                continue
            age_years = float(age_dt[i]) / n_dt_per_year
            indices[i] = self.get_index(csv_name, age_years, role)

        return indices


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------

_THRESHOLD_RE = re.compile(r"^(.+?)\s*<\s*([\d.]+)\s*$")


def _parse_label(label: str) -> tuple[str, float]:
    """Parse a label like "cod < 0.4" → ("cod", 0.4) or "cod" → ("cod", inf)."""
    label = label.strip()
    m = _THRESHOLD_RE.match(label)
    if m:
        return m.group(1).strip(), float(m.group(2))
    return label, float("inf")


def _parse_labels(labels: list[str]) -> dict[str, list[StageInfo]]:
    """Parse a list of labels into per-species stage lookups.

    Returns dict mapping species_name → sorted list of StageInfo.
    Stages are sorted by threshold ascending (threshold < inf comes first).
    """
    # Collect (species_name, threshold, index) tuples
    entries: dict[str, list[tuple[float, int]]] = {}
    for idx, label in enumerate(labels):
        name, threshold = _parse_label(label)
        entries.setdefault(name, []).append((threshold, idx))

    # Sort each species' stages by threshold (finite thresholds first)
    result: dict[str, list[StageInfo]] = {}
    for name, stage_list in entries.items():
        stage_list.sort(key=lambda t: (t[0] == float("inf"), t[0]))
        result[name] = [StageInfo(threshold=t, matrix_index=i) for t, i in stage_list]

    return result
