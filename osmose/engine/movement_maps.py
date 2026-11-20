"""MovementMapSet: CSV spatial probability map loading and index_maps construction.

Manages the set of movement maps for a single species, providing a lookup table
from (age_dt, step) to the corresponding probability grid. This is the data
infrastructure for map-based (B1) movement — no movement logic is implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from osmose.engine.path_resolution import resolve_data_path

logger = logging.getLogger(__name__)


def _resolve_path(filepath_str: str, config_dir: str = "") -> Path:
    """Resolve a relative CSV map file path against multiple candidate directories.

    Thin wrapper around :func:`resolve_data_path`. Missing explicit movement
    map files are config errors: returning ``None`` is reserved for authored
    ``movement.file.mapN=null`` entries that intentionally move schools out.
    """
    result = resolve_data_path(filepath_str, config_dir=config_dir)
    if result is not None:
        return result
    raise FileNotFoundError(f"Movement map file could not be resolved: {filepath_str!r}")


def _resolve_pattern(value: str, config_dir: str, key: str) -> Path:
    """The directory and file-name pattern of a NetCDF map file (Java ``getFile`` resolves the
    path; the name is then a regular expression over that directory's files)."""
    value = value.strip()
    if not value or value.lower() == "null":
        raise ValueError(f"movement.netcdf.enabled needs {key}")
    path = Path(value)
    if not path.is_absolute() and config_dir:
        path = Path(config_dir) / path
    return path


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
        # Fast path: NumPy's str -> float64 equals float() part by part (whitespace, signs,
        # exponents, 'nan', 'inf'; checked on 200,000 strings). It raises on 'na', an empty field
        # or anything float() rejects, which then takes the original per-part rule below.
        try:
            row_values: NDArray[np.float64] | list[float] = np.array(parts, dtype=np.float64)
        except ValueError:
            row_values = []
            for part in parts:
                p = part.strip().lower()
                if p in ("na", "nan", ""):
                    row_values.append(np.nan)
                else:
                    row_values.append(float(p))
        if len(row_values) < nx:
            raise ValueError(
                f"Movement map CSV row {csv_row_idx} has {len(row_values)} columns, "
                f"expected {nx} (grid nx)"
            )
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
        strict: bool = False,
        ocean_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Build index_maps and load CSV grids for one species.

        With ``movement.netcdf.enabled = true`` the maps come from NetCDF files instead
        (:meth:`_load_netcdf`); ``ocean_mask`` then decides which time steps are empty.

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
                suffix = key[len("movement.species.map") :]
                if suffix.isdigit() and val.strip().lower() == species_name.strip().lower():
                    matched_map_numbers.append(int(suffix))

        matched_map_numbers.sort()

        if str(config.get("movement.netcdf.enabled", "")).strip().lower() == "true":
            self._load_netcdf(
                config,
                species_name,
                matched_map_numbers,
                n_dt_per_year,
                n_total_steps,
                lifespan_dt,
                ny,
                nx,
                config_dir,
                strict,
                ocean_mask,
            )
            return

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
            default_last = str(lifespan_dt / n_dt_per_year)
            last_age_raw = float(
                config.get(
                    f"movement.lastage.map{n}",
                    default_last,
                )
            )
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
            # Every (age, year, season) cell of this map gets the same value, and later maps
            # overwrite earlier ones, so one fancy assignment per map equals the per-cell loop
            # (negative indices wrap the same way in both). The loop was ~1.2M Python iterations
            # on EEC at 10 years, paid by simulate() and by the C# bundle build alike.
            ages = np.arange(init_age_dt, last_age_dt + 1)
            ages = ages[ages < lifespan_dt]
            cols = np.array([y * n_dt_per_year + s for y in years for s in steps], dtype=np.int64)
            cols = cols[cols < n_total_steps]
            if ages.size and cols.size:
                self.index_maps[np.ix_(ages, cols)] = remap[map_idx]

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
                    raise ValueError(f"Failed to load movement map file {fp}: {exc}") from exc

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
        uncovered = int((self.index_maps == -1).sum())
        if uncovered > 0:
            msg = (
                f"Species {species_name!r}: {uncovered} of {lifespan_dt * n_total_steps} "
                f"(age_dt, step) slots have no movement map assigned"
            )
            if strict:
                raise ValueError(msg)
            logger.warning("%s", msg)

    def _load_netcdf(
        self,
        config: dict[str, str],
        species_name: str,
        map_numbers: list[int],
        n_dt_per_year: int,
        n_total_steps: int,
        lifespan_dt: int,
        ny: int,
        nx: int,
        config_dir: str,
        strict: bool,
        ocean_mask: NDArray[np.bool_] | None,
    ) -> None:
        """Java ``MapSet.loadMapsNcMaps`` (v4.4.1): every time step of map k's variable is a map.

        Map k reads ``movement.variable.mapk`` from the files matching ``movement.file.mapk``
        (a regular expression on the file name, the matches sorted and concatenated in time;
        a plain name matches itself), with ``movement.nsteps.year.mapk`` steps per year. A step
        whose ocean cells sum to 0 is an empty map (the school goes out of the domain). For ages
        ``movement.initialAge`` to ``movement.lastAge`` (capped at the lifespan), simulation step
        t uses file step ``(t // (n_dt_per_year // nsteps)) % n_file_steps`` (ForcingFile
        .getNcStep); a later map number overrides an earlier one. Rows are taken as stored, with
        no flip (only CSV maps are written north first). ``movement.steps`` / ``years`` are not
        read, as in Java.
        """
        import re

        from osmose.engine._netcdf import open_dataset_safe

        self.maps = []
        for n in map_numbers:
            file_key = f"movement.file.map{n}"
            pattern_path = _resolve_pattern(config.get(file_key, ""), config_dir, file_key)
            regex = re.compile(pattern_path.name)
            files = sorted(p for p in pattern_path.parent.iterdir() if regex.fullmatch(p.name))
            if not files:
                raise FileNotFoundError(
                    f"{file_key}: no file in {pattern_path.parent} matches {pattern_path.name!r}"
                )
            var = config.get(f"movement.variable.map{n}", "").strip()
            if not var:
                raise ValueError(f"movement.netcdf.enabled needs movement.variable.map{n}")
            nc_per_year = int(config.get(f"movement.nsteps.year.map{n}", "0") or 0)
            if nc_per_year <= 0 or n_dt_per_year // nc_per_year == 0:
                raise ValueError(
                    f"movement.nsteps.year.map{n} = {nc_per_year}: Java maps a step to "
                    f"step // (ndtperyear // nsteps.year), which divides by zero unless it is "
                    f"between 1 and simulation.time.ndtperyear ({n_dt_per_year})"
                )
            chunks = []
            for f in files:
                values = np.asarray(open_dataset_safe(f)[var].values, dtype=np.float64)
                if values.ndim == 4:  # (time, z, y, x): Java takes the first level
                    values = values[:, 0]
                if values.ndim == 2:
                    values = values[np.newaxis]
                if values.shape[1:] != (ny, nx):
                    raise ValueError(
                        f"{f}:{var} has a {values.shape[1:]} grid; the model grid is {(ny, nx)}"
                    )
                chunks.append(values)
            stack = np.concatenate(chunks, axis=0)
            base = len(self.maps)
            for grid in stack:
                total = float(grid[ocean_mask].sum()) if ocean_mask is not None else grid.sum()
                self.maps.append(None if total == 0.0 else grid.copy())

            init_age_dt = round(
                float(config.get(f"movement.initialage.map{n}", "0")) * n_dt_per_year
            )
            last_age_raw = float(
                config.get(f"movement.lastage.map{n}", str(lifespan_dt / n_dt_per_year))
            )
            last_age_dt = min(round(last_age_raw * n_dt_per_year), lifespan_dt - 1)
            ages = np.arange(init_age_dt, last_age_dt + 1)
            ages = ages[(ages >= 0) & (ages < lifespan_dt)]
            steps = np.arange(n_total_steps)
            file_step = (steps // (n_dt_per_year // nc_per_year)) % len(stack)
            if ages.size:
                self.index_maps[ages[:, None], steps[None, :]] = base + file_step[None, :]

        self._eliminate_twin_maps(ocean_mask)
        self.n_maps = len(self.maps)
        self.max_proba = np.zeros(self.n_maps, dtype=np.float64)
        for i, grid in enumerate(self.maps):
            if grid is not None:
                max_val = float(np.nanmax(grid))
                self.max_proba[i] = 0.0 if max_val >= 1.0 else max_val

        uncovered = int((self.index_maps == -1).sum())
        if uncovered > 0:
            msg = (
                f"Species {species_name!r}: {uncovered} of {self.index_maps.size} "
                f"(age_dt, step) slots have no movement map assigned"
            )
            if strict:
                raise ValueError(msg)
            logger.warning("%s", msg)

    def _eliminate_twin_maps(self, ocean_mask: NDArray[np.bool_] | None) -> None:
        """Java ``MapSet.eliminateTwinMapNC``: a map equal to an earlier one takes its index.

        Equality is Java ``GridMap.equals`` on what it stores: ocean cells as float32 (land is a
        fixed value there), NaN equal to NaN. Empty (None) maps keep their own index. Sharing the
        index matters to movement: a school walks while its map index stays the same and is
        placed afresh when it changes, so two identical frames in a row must not re-place it.
        """

        def stored(g: NDArray[np.float64]) -> NDArray[np.float32]:
            return (g[ocean_mask] if ocean_mask is not None else g).astype(np.float32)

        twin = list(range(len(self.maps)))
        for k, grid_k in enumerate(self.maps):
            if grid_k is None:
                continue
            for j in range(k - 1, -1, -1):
                grid_j = self.maps[j]
                if grid_j is not None and np.array_equal(
                    stored(grid_k), stored(grid_j), equal_nan=True
                ):
                    twin[k] = twin[j]
                    self.maps[k] = None
                    break
        lookup = np.asarray(twin, dtype=np.int32)
        valid = self.index_maps >= 0
        self.index_maps[valid] = lookup[self.index_maps[valid]]

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
