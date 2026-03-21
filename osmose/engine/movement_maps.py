"""MovementMapSet: CSV spatial probability map loading and index_maps construction.

Manages the set of movement maps for a single species, providing a lookup table
from (age_dt, step) to the corresponding probability grid. This is the data
infrastructure for map-based (B1) movement — no movement logic is implemented here.
"""

from __future__ import annotations

import glob as _glob
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _resolve_path(filepath_str: str, config_dir: str = "") -> Path:
    """Resolve a relative CSV map file path against multiple candidate directories.

    Tries (in order):
    1. The path as-is (works for absolute paths or paths relative to CWD).
    2. Relative to the config directory (if provided).
    3. Relative to ``data/examples/``.
    4. Relative to any ``data/*/`` subdirectory (sorted for determinism).

    Returns the first existing candidate, or the original path if none found
    (so that the subsequent ``open()`` call will raise a clear ``FileNotFoundError``).
    """
    p = Path(filepath_str)
    if p.exists():
        return p
    search_dirs: list[Path] = []
    if config_dir:
        search_dirs.append(Path(config_dir))
    search_dirs.append(Path("data/examples"))
    search_dirs.extend(Path(d) for d in sorted(_glob.glob("data/*/")))
    for base in search_dirs:
        candidate = base / p
        if candidate.exists():
            return candidate
    return p  # fall through — open() will raise FileNotFoundError with the original path


def _parse_semicolon_ints(value: str, limit: int) -> list[int]:
    """Parse a semicolon-separated string of ints, filtering values >= limit."""
    return [int(v.strip()) for v in value.split(";") if v.strip() and int(v.strip()) < limit]


def _load_csv_grid(path: Path, ny: int, nx: int) -> NDArray[np.float64]:
    """Load a semicolon-delimited 2D grid CSV file.

    Row 0 in CSV = northernmost row = grid row ny-1 (flipped on load).
    Values: -99 = land/absent (stored as-is). 'na'/'nan' strings -> np.nan.
    """
    grid = np.empty((ny, nx), dtype=np.float64)
    with open(path) as f:
        lines = f.readlines()

    for csv_row_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        row_values: list[float] = []
        for part in parts:
            p = part.strip().lower()
            if p in ("na", "nan", ""):
                row_values.append(np.nan)
            else:
                row_values.append(float(p))
        grid_row = ny - 1 - csv_row_idx
        grid[grid_row, :] = row_values[:nx]

    return grid


class MovementMapSet:
    """Spatial movement map set for one species.

    Attributes
    ----------
    index_maps : NDArray[np.int32]
        Shape (lifespan_dt, n_total_steps). Value is the index into ``maps``
        for that age/step combination, or -1 if no map is assigned.
    maps : list[NDArray[np.float64] | None]
        2D grids of shape (ny, nx). None indicates a 'null' (out-of-domain) map.
    max_proba : NDArray[np.float64]
        Per-map maximum cell value. 0.0 for presence/absence maps (max >= 1.0),
        actual nanmax otherwise.
    n_maps : int
        Number of map entries (before deduplication).
    """

    index_maps: NDArray[np.int32]
    maps: list[NDArray[np.float64] | None]
    max_proba: NDArray[np.float64]
    n_maps: int

    def __init__(
        self,
        config: dict[str, str],
        species_name: str,
        n_dt_per_year: int,
        n_years: int,
        lifespan_dt: int,
        ny: int,
        nx: int,
        config_dir: str = "",
    ) -> None:
        """Build index_maps and load CSV grids for one species.

        Parameters
        ----------
        config:
            Flat OSMOSE config dict (keys already lowercased).
        species_name:
            Species name as it appears in ``movement.species.map{N}`` values.
        n_dt_per_year:
            Number of time steps per year.
        n_years:
            Total simulation years.
        lifespan_dt:
            Maximum age in time steps.
        ny, nx:
            Grid dimensions.
        """
        n_total_steps = max(n_years * n_dt_per_year, n_dt_per_year)

        self.index_maps = np.full((lifespan_dt, n_total_steps), -1, dtype=np.int32)

        # Collect all map numbers whose species matches (case-insensitive)
        matched_map_numbers: list[int] = []
        for key, val in config.items():
            if key.startswith("movement.species.map"):
                suffix = key[len("movement.species.map"):]
                if suffix.isdigit() and val.strip().lower() == species_name.strip().lower():
                    matched_map_numbers.append(int(suffix))

        matched_map_numbers.sort()

        # Per-map metadata: (map_idx, file_path_or_none)
        map_file_paths: list[Path | None] = []
        raw_grids: list[NDArray[np.float64] | None] = []

        # Track file-path -> first assigned index for deduplication
        canonical_index: dict[str, int] = {}
        # Remap table: local map index -> canonical index
        remap: dict[int, int] = {}

        for n in matched_map_numbers:
            map_idx = len(map_file_paths)

            # --- Age range ---
            init_age_raw = float(config.get(f"movement.initialage.map{n}", "0"))
            last_age_raw = float(config.get(f"movement.lastage.map{n}", str(lifespan_dt / n_dt_per_year)))
            init_age_dt = round(init_age_raw * n_dt_per_year)
            last_age_dt = min(round(last_age_raw * n_dt_per_year), lifespan_dt - 1)

            # --- Steps (seasons within a year) ---
            steps_raw = config.get(f"movement.steps.map{n}", "")
            if steps_raw and steps_raw.strip().lower() not in ("null", ""):
                steps = _parse_semicolon_ints(steps_raw, n_dt_per_year)
            else:
                steps = list(range(n_dt_per_year))

            # --- Year range ---
            years_raw = config.get(f"movement.years.map{n}", "")
            if years_raw and years_raw.strip().lower() not in ("null", ""):
                years = _parse_semicolon_ints(years_raw, n_years)
            else:
                init_year = int(config.get(f"movement.initialyear.map{n}", "0"))
                last_year = int(config.get(f"movement.lastyear.map{n}", str(n_years - 1)))
                # last_year is INCLUSIVE; filter years >= n_years
                years = [y for y in range(init_year, last_year + 1) if y < n_years]

            # --- File ---
            file_val = config.get(f"movement.file.map{n}", "null")
            if file_val.strip().lower() == "null":
                file_path = None
            else:
                file_path = _resolve_path(file_val, config_dir)

            # --- Deduplication ---
            path_key = str(file_path) if file_path is not None else None
            if path_key is not None and path_key in canonical_index:
                canonical = canonical_index[path_key]
                remap[map_idx] = canonical
                # Still need a placeholder so list indices stay consistent
                map_file_paths.append(file_path)
                raw_grids.append(None)  # duplicate — will be replaced by remap
            else:
                if path_key is not None:
                    canonical_index[path_key] = map_idx
                remap[map_idx] = map_idx
                map_file_paths.append(file_path)
                raw_grids.append(None)  # loaded below

            # --- Fill index_maps ---
            for age_dt in range(init_age_dt, last_age_dt + 1):
                if age_dt >= lifespan_dt:
                    continue
                for year in years:
                    for season in steps:
                        global_step = year * n_dt_per_year + season
                        if global_step >= n_total_steps:
                            continue
                        self.index_maps[age_dt, global_step] = remap[map_idx]

        # --- Load CSV grids ---
        for i, fp in enumerate(map_file_paths):
            if remap.get(i, i) != i:
                # Duplicate: reuse the canonical grid
                continue
            if fp is None:
                raw_grids[i] = None
            else:
                try:
                    raw_grids[i] = _load_csv_grid(fp, ny, nx)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    logger.error("Failed to load movement map file %s: %s", fp, exc)
                    raw_grids[i] = None

        # --- Build deduplicated maps list ---
        # Collect only canonical indices (remap[i] == i), preserving order
        canonical_indices_ordered: list[int] = []
        seen: set[int] = set()
        for i in range(len(map_file_paths)):
            c = remap.get(i, i)
            if c not in seen:
                seen.add(c)
                canonical_indices_ordered.append(c)

        # Build final maps and max_proba; also build a remapping from old canonical idx
        # to position in final list
        old_to_new: dict[int, int] = {}
        self.maps = []
        for new_pos, old_idx in enumerate(canonical_indices_ordered):
            old_to_new[old_idx] = new_pos
            self.maps.append(raw_grids[old_idx])

        # --- max_proba ---
        n_canonical = len(self.maps)
        self.max_proba = np.zeros(n_canonical, dtype=np.float64)
        for new_pos, grid in enumerate(self.maps):
            if grid is None:
                self.max_proba[new_pos] = 0.0
            else:
                max_val = float(np.nanmax(grid))
                if max_val >= 1.0:
                    self.max_proba[new_pos] = 0.0  # presence/absence: uniform acceptance
                else:
                    self.max_proba[new_pos] = max_val

        # --- Remap index_maps to new positions ---
        if old_to_new:
            # Vectorised remap: build a lookup array
            max_old = max(old_to_new.keys()) + 1
            lookup = np.full(max_old + 1, -1, dtype=np.int32)
            for old, new in old_to_new.items():
                lookup[old] = new
            # Apply: only remap non-(-1) entries
            valid = self.index_maps >= 0
            self.index_maps[valid] = lookup[self.index_maps[valid]]

        self.n_maps = n_canonical

        # --- Validate: warn about uncovered (age, step) slots ---
        for age_dt in range(lifespan_dt):
            for step in range(n_total_steps):
                if self.index_maps[age_dt, step] == -1:
                    logger.warning(
                        "No movement map for species=%r age_dt=%d step=%d",
                        species_name,
                        age_dt,
                        step,
                    )

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def get_map(self, age_dt: int, step: int) -> NDArray[np.float64] | None:
        """Return the probability grid for (age_dt, step), or None if absent."""
        if age_dt < 0 or age_dt >= self.index_maps.shape[0]:
            return None
        if step < 0 or step >= self.index_maps.shape[1]:
            return None
        idx = int(self.index_maps[age_dt, step])
        if idx == -1:
            return None
        return self.maps[idx]

    def get_index(self, age_dt: int, step: int) -> int:
        """Return the map index for (age_dt, step), or -1 if absent."""
        if age_dt < 0 or age_dt >= self.index_maps.shape[0]:
            return -1
        if step < 0 or step >= self.index_maps.shape[1]:
            return -1
        return int(self.index_maps[age_dt, step])
