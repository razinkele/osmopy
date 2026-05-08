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
class _PerSpeciesStages:
    """Vectorised per-species stage cache used by compute_school_indices.

    Built once at AccessibilityMatrix construction time. Replaces the
    per-call int→string→stages-list traversal with a single searchsorted
    on float64 threshold arrays.
    """

    thresholds: NDArray[np.float64]   # sorted ascending; last element is +inf for open-ended labels
    matrix_indices: NDArray[np.int32]  # same length, parallel to thresholds


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
    # Vectorised per-species stage cache, keyed by int sp_idx (NOT csv-label
    # string). Built at from_csv time via _build_stages_by_role; consumed by
    # compute_school_indices for the searchsorted fast path. Schools whose
    # species_id is absent from a role's dict keep the -1 sentinel — matches
    # the legacy loop's `resolved.get(sp) is None: continue` skip.
    _stages_by_role: dict[str, dict[int, _PerSpeciesStages]] = field(
        default_factory=lambda: {"prey": {}, "pred": {}}
    )

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

        # M5: accessibility coefficients should fall in [0, 1] — they are
        # consumption-fraction multipliers. Values > 1 yield more biomass
        # consumed than the prey-school holds; biomass-conservation breaks.
        # Surface as a warning rather than raising — calibration sometimes
        # pushes coefficients above 1 transiently while exploring the
        # parameter space.
        if (raw_matrix > 1.0).any():
            n_violations = int((raw_matrix > 1.0).sum())
            max_val = float(raw_matrix.max())
            import warnings
            warnings.warn(
                f"predation accessibility matrix at {csv_path}: {n_violations} "
                f"coefficient(s) exceed 1.0 (max={max_val:.3f}); biomass "
                f"conservation may be violated. Re-check the CSV.",
                stacklevel=2,
            )

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

        instance = cls(
            raw_matrix=raw_matrix,
            prey_lookup=prey_lookup,
            pred_lookup=pred_lookup,
            _species_name_map=name_map,
        )
        instance._build_stages_by_role(species_names)
        return instance

    def _build_stages_by_role(self, species_names: list[str]) -> None:
        """Translate string-keyed prey/pred lookups into int-keyed numpy arrays.

        Iterates `species_names` in sp_idx order, resolves each name via
        `_species_name_map`, and inserts a `_PerSpeciesStages` entry under
        each role only when the lookup contains a non-empty stages list.
        Species whose name does not resolve are absent from both role
        dicts — `compute_school_indices` then leaves their schools at -1.
        """
        self._stages_by_role = {"prey": {}, "pred": {}}
        for sp_idx, name in enumerate(species_names):
            csv_name = self.resolve_name(name)
            if csv_name is None:
                continue
            for role, lookup in (("prey", self.prey_lookup), ("pred", self.pred_lookup)):
                stages = lookup.get(csv_name)
                if not stages:
                    continue
                thresholds = np.array(
                    [s.threshold for s in stages], dtype=np.float64
                )
                matrix_indices = np.array(
                    [s.matrix_index for s in stages], dtype=np.int32
                )
                # Construction-time invariant: every cached species has at
                # least one stage. Without this, the searchsorted+clamp
                # path would index `matrix_indices[-1]` and silently wrap
                # to a wrong row.
                assert len(thresholds) >= 1, (
                    f"AccessibilityMatrix: {role} stages for {name!r} "
                    f"(csv={csv_name!r}) must be non-empty"
                )
                self._stages_by_role[role][sp_idx] = _PerSpeciesStages(
                    thresholds=thresholds,
                    matrix_indices=matrix_indices,
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

        Vectorised over schools via per-species `np.searchsorted` against
        precomputed threshold arrays (`_stages_by_role`, built at from_csv
        time). For the equivalent loop implementation kept for cross-check,
        see `_compute_school_indices_loop`.

        Parameters
        ----------
        species_id:
            Species index per school.
        age_dt:
            Age in timesteps per school.
        n_dt_per_year:
            Timesteps per year (for age conversion).
        all_species_names:
            Full species name list. Retained for API compatibility with
            the loop implementation; the vectorised path consults the
            int-keyed `_stages_by_role` cache built at construction time
            from the same `species_names` argument passed to `from_csv`.
        role:
            "prey" or "pred".

        Returns
        -------
        Array of matrix indices, shape (n_schools,). -1 if not found.
        """
        indices = np.full(species_id.shape, -1, dtype=np.int32)
        if species_id.size == 0:
            return indices
        age_years = age_dt.astype(np.float64) / n_dt_per_year
        stages_by_sp = self._stages_by_role.get(role, {})
        for sp_idx, stages in stages_by_sp.items():
            mask = species_id == sp_idx
            if not mask.any():
                continue
            # searchsorted-right + clamp reproduces the legacy loop's
            #   `if age < threshold: return matrix_index` semantics, with
            #   `return stages[-1]` as the fallback for ages >= all
            #   thresholds. The clamp is also sufficient if the last
            #   threshold is finite (no `+inf` sentinel) — searchsorted
            #   returns len(thresholds), the clamp drops it to len-1.
            bin_idx = np.searchsorted(
                stages.thresholds, age_years[mask], side="right"
            )
            np.minimum(bin_idx, len(stages.thresholds) - 1, out=bin_idx)
            indices[mask] = stages.matrix_indices[bin_idx]
        return indices

    def _compute_school_indices_loop(
        self,
        species_id: NDArray[np.int32],
        age_dt: NDArray[np.int32],
        n_dt_per_year: int,
        all_species_names: list[str],
        role: str = "prey",
    ) -> NDArray[np.int32]:
        """Reference loop implementation kept for cross-check tests.

        Behaviourally equivalent to `compute_school_indices`. NOT used in
        production — exists only so parity tests can call both
        implementations against the same inputs and assert element-wise
        equality.
        """
        n = len(species_id)
        indices = np.full(n, -1, dtype=np.int32)
        resolved: dict[int, str | None] = {}
        for sp_idx in range(len(all_species_names)):
            resolved[sp_idx] = self.resolve_name(all_species_names[sp_idx])
        for i in range(n):
            sp = species_id[i]
            csv_name = resolved.get(int(sp))
            if csv_name is None:
                continue
            age_years_i = float(age_dt[i]) / n_dt_per_year
            indices[i] = self.get_index(csv_name, age_years_i, role)
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
