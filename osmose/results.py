"""Read OSMOSE simulation output files."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from osmose.engine.output import (
    _build_bioen_dataframes,
    _build_diet_dataframe,
    _build_distribution_dataframes,
    _build_mortality_dataframes,
    _build_species_dataframes,
    _build_yield_dataframes,
)
from osmose.logging import setup_logging

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import StepOutput

_log = setup_logging("osmose.results")

# Module-level cache for preamble detection (path -> (mtime_ns, size, n_preamble_lines))
_PREAMBLE_CACHE: dict[Path, tuple[int, int, int]] = {}

# Engine-defined output subdirectories (searched in addition to root)
_ENGINE_SUBDIRS = ("Mortality", "Bioen")

# Regex to strip _SimuN suffix from filenames
_SIMU_SUFFIX_RE = re.compile(r"_Simu\d+$")


def _find_output_files(output_dir: Path, pattern: str) -> list[Path]:
    """Search root + known engine subdirectories for output files."""
    results: list[Path] = []
    results.extend(sorted(output_dir.glob(pattern)))
    for subdir in _ENGINE_SUBDIRS:
        sub_path = output_dir / subdir
        if sub_path.is_dir():
            results.extend(sorted(sub_path.glob(pattern)))
    return results


def _detect_preamble_lines(path: Path) -> int:
    """Detect number of non-tabular preamble lines before the CSV header.

    Engine and Java CSVs may prepend 1–2 description lines before the actual
    header row.  We find the header by looking for the first line whose field
    count matches the next line's field count (i.e. two consecutive rows that
    parse to the same width).

    Results are cached by (path, mtime_ns, size) to avoid re-parsing.
    """
    resolved = path.resolve()
    st = path.stat()
    cache_key = (st.st_mtime_ns, st.st_size)
    cached = _PREAMBLE_CACHE.get(resolved)
    if cached is not None and (cached[0], cached[1]) == cache_key:
        return cached[2]

    raw_lines: list[str] = []
    with open(path, "r", newline="") as fh:
        for _ in range(10):  # only need to inspect first few lines
            line = fh.readline()
            if not line:
                break
            raw_lines.append(line)

    if len(raw_lines) < 2:
        result = 0
    else:

        def _field_count(line: str) -> int:
            reader = csv.reader(io.StringIO(line))
            row = next(reader, [])
            return len(row)

        counts = [_field_count(ln) for ln in raw_lines]
        result = 0
        for i in range(len(counts) - 1):
            if counts[i] == counts[i + 1] and counts[i] > 1:
                result = i
                break
        else:
            # Fallback: if no equal-width pair found (e.g. header-only file),
            # use the first line with >1 fields as the header row.
            for i, c in enumerate(counts):
                if c > 1:
                    result = i
                    break

    _PREAMBLE_CACHE[resolved] = (st.st_mtime_ns, st.st_size, result)
    return result


def _read_output_csv(path: Path) -> pd.DataFrame:
    """Read an engine/Java CSV, auto-skipping any preamble lines."""
    skip = _detect_preamble_lines(path)
    return pd.read_csv(path, skiprows=skip)


def _matches_output_type(stem: str, output_type: str, prefix: str) -> bool:
    """Check if a filename stem matches exactly the given output type.

    Prevents 'biomass' from matching 'biomassByAge', 'biomassBySize', etc.
    After the prefix_{type}, the next char must be '_', '-', or end of stem
    (before _SimuN).
    """
    base = _SIMU_SUFFIX_RE.sub("", stem)
    expected = f"{prefix}_{output_type}"
    if not base.startswith(expected):
        return False
    remainder = base[len(expected):]
    # Exact match (all-species) or separator before species name
    return remainder == "" or remainder[0] in ("_", "-")


def _extract_species(stem: str, output_type: str, prefix: str) -> str | None:
    """Extract species name from an output filename stem.

    Returns the species name, or None for all-species aggregate files.

    Handles patterns:
      - {prefix}_{type}_Simu0          → None (all-species)
      - {prefix}_{type}_{species}_Simu0  → species
      - {prefix}_{type}-{species}_Simu0  → species (mortality dash naming)
    """
    # Strip _SimuN suffix
    base = _SIMU_SUFFIX_RE.sub("", stem)
    # Remove prefix
    expected_prefix = f"{prefix}_{output_type}"
    if not base.startswith(expected_prefix):
        return stem  # fallback: return full stem
    remainder = base[len(expected_prefix):]
    if not remainder:
        return None  # all-species file
    # remainder starts with '_' or '-' followed by species name
    if remainder[0] in ("_", "-"):
        return remainder[1:]
    return stem  # fallback


def _melt_wide_to_long(
    wide_df: pd.DataFrame,
    species: str | None = None,
) -> pd.DataFrame:
    """Melt a wide-form DataFrame (Time + bin columns) into long-form.

    Returns columns: time, species, bin, value.
    If ``wide_df`` has a ``species`` column, it is preserved; otherwise a
    ``species`` column is added with the value ``'all'``.
    If ``species`` is provided, filter the result to that species name.
    """
    time_col = wide_df.columns[0]
    has_species = "species" in wide_df.columns
    id_cols = [time_col] + (["species"] if has_species else [])
    bin_cols = [c for c in wide_df.columns[1:] if c != "species"]

    melted = wide_df.melt(
        id_vars=id_cols,
        value_vars=bin_cols,
        var_name="bin",
        value_name="value",
    )
    melted = melted.rename(columns={time_col: "time"})
    if not has_species:
        melted["species"] = "all"
    melted = melted[["time", "species", "bin", "value"]]

    if species is not None:
        melted = melted[melted["species"] == species]
    return melted


# Output-type keys that represent cross-species (single-file) outputs — their
# disk-shape keys do not embed a species name.
_CROSS_SPECIES_OUTPUT_TYPES = {"biomass", "abundance", "yield", "dietMatrix"}


def _build_dataframes_from_outputs(
    outputs: list[StepOutput],
    config: EngineConfig,
    grid: Grid,
) -> dict[str, pd.DataFrame]:
    """Build the ``OsmoseResults._csv_cache`` payload from in-memory StepOutputs.

    Returns a dict mapping ``output_type`` (e.g., ``"biomass"``,
    ``"biomassByAge"``) to a LONG-form DataFrame with an added ``species``
    column — exactly the shape ``_read_species_output`` builds when reading
    per-species CSV files from disk.

    Two steps:
      1. Call each ``_build_*_dataframes`` helper to get disk-shape dicts.
      2. Adapt to cache shape: group by output_type, add species column,
         concatenate same-output_type entries.

    The ``grid`` argument is reserved for future spatial/NetCDF in-memory work.
    """
    disk_shape: dict[str, pd.DataFrame] = {}
    disk_shape.update(_build_species_dataframes(outputs, config))
    disk_shape.update(_build_mortality_dataframes(outputs, config))
    disk_shape.update(_build_yield_dataframes(outputs, config))
    disk_shape.update(_build_distribution_dataframes(outputs, config))
    if config.bioen_enabled:
        disk_shape.update(_build_bioen_dataframes(outputs, config))
    if config.diet_output_enabled:
        disk_shape.update(_build_diet_dataframe(outputs, config))
    _ = grid  # reserved for future NetCDF-in-memory work

    cache_shape: dict[str, list[pd.DataFrame]] = {}
    for key, df in disk_shape.items():
        if key in _CROSS_SPECIES_OUTPUT_TYPES:
            output_type, sp_name = key, "all"
        else:
            output_type, _, sp_name = key.partition("_")
        annotated = df.copy()
        annotated["species"] = sp_name
        cache_shape.setdefault(output_type, []).append(annotated)

    return {
        ot: pd.concat(frames, ignore_index=True)
        for ot, frames in cache_shape.items()
    }


class OsmoseResults:
    """Read and query OSMOSE simulation outputs.

    OSMOSE writes output as CSV and/or NetCDF files. This class provides
    a unified interface to access biomass, abundance, yield, diet, and
    mortality data.
    """

    def __init__(self, output_dir: Path, prefix: str = "osm", strict: bool = True):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.strict = strict
        self._nc_cache: dict[str, xr.Dataset] = {}
        self._csv_cache: dict[str, pd.DataFrame] = {}
        self._in_memory: bool = False

    @classmethod
    def from_outputs(
        cls,
        outputs: list[StepOutput],
        engine_config: EngineConfig,
        grid: Grid,
        *,
        prefix: str = "osm",
    ) -> OsmoseResults:
        """Construct an ``OsmoseResults`` backed by in-memory StepOutputs.

        No disk I/O. Supports every CSV-backed getter (biomass, abundance,
        yield_biomass, mortality, diet_matrix, mean_size, mean_trophic_level,
        biomass_by_age, biomass_by_size, biomass_by_tl, abundance_by_age, etc.).
        Getters that return an ``xr.Dataset`` from NetCDF (spatial_biomass,
        read_netcdf) raise ``FileNotFoundError`` in this mode.

        Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
        """
        obj = cls.__new__(cls)
        obj.output_dir = None  # type: ignore[assignment]
        obj.prefix = prefix
        obj.strict = True
        obj._csv_cache = _build_dataframes_from_outputs(outputs, engine_config, grid)
        # __init__ initializes _nc_cache = {} for NetCDF caching; since we're
        # bypassing __init__ via __new__, set it explicitly. close_cache()
        # iterates _nc_cache and would raise AttributeError otherwise.
        obj._nc_cache = {}
        obj._in_memory = True
        return obj

    def __enter__(self) -> OsmoseResults:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_cache()

    def _raise_if_strict(self, pattern: str) -> None:
        """Raise FileNotFoundError in strict mode when no files match."""
        if not self.strict:
            return
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory does not exist: {self.output_dir}")
        raise FileNotFoundError(f"No files matching '{pattern}' in {self.output_dir}")

    def list_outputs(self) -> list[str]:
        """List all output files in the output directory and engine subdirs."""
        if self._in_memory:
            return sorted(self._csv_cache.keys())
        if not self.output_dir.exists():
            return []
        files = []
        for f in sorted(self.output_dir.iterdir()):
            if f.suffix in (".csv", ".nc"):
                files.append(f.name)
        for subdir in _ENGINE_SUBDIRS:
            sub_path = self.output_dir / subdir
            if sub_path.is_dir():
                for f in sorted(sub_path.iterdir()):
                    if f.suffix in (".csv", ".nc"):
                        files.append(f"{subdir}/{f.name}")
        return files

    def read_csv(self, pattern: str) -> dict[str, pd.DataFrame]:
        """Read CSV output files matching a glob pattern."""
        if self._in_memory:
            raise FileNotFoundError(
                f"In-memory OsmoseResults does not support read_csv "
                f"(requested: {pattern}). Use a disk-backed OsmoseResults "
                f"or call the specific getter (biomass(), abundance(), etc.)."
            )
        result = {}
        for f in _find_output_files(self.output_dir, pattern):
            try:
                result[f.stem] = _read_output_csv(f)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                _log.warning("Skipping malformed CSV %s: %s", f.name, exc)
        return result

    def read_netcdf(self, filename: str) -> xr.Dataset:
        """Read a NetCDF output file, with caching."""
        if self._in_memory:
            raise FileNotFoundError(
                f"In-memory OsmoseResults does not support NetCDF outputs "
                f"(requested: {filename}). Use the disk-backed OsmoseResults "
                f"constructor if you need spatial NetCDF outputs."
            )
        if filename not in self._nc_cache:
            path = self.output_dir / filename
            self._nc_cache[filename] = xr.open_dataset(path)
        return self._nc_cache[filename]

    def biomass(self, species: str | None = None) -> pd.DataFrame:
        """Read biomass time series.

        Returns DataFrame with columns: time, species, biomass.
        Reads from CSV files matching *biomass*.csv pattern.
        """
        return self._read_species_output("biomass", species)

    def abundance(self, species: str | None = None) -> pd.DataFrame:
        """Read abundance time series."""
        return self._read_species_output("abundance", species)

    def yield_biomass(self, species: str | None = None) -> pd.DataFrame:
        """Read yield/catch biomass time series."""
        return self._read_species_output("yield", species)

    def mortality(self, species: str | None = None) -> pd.DataFrame:
        """Read mortality breakdown."""
        return self._read_species_output("mortalityRate", species)

    def diet_matrix(self, species: str | None = None) -> pd.DataFrame:
        """Read diet composition matrix."""
        return self._read_species_output("dietMatrix", species)

    def mean_size(self, species: str | None = None) -> pd.DataFrame:
        """Read mean size time series."""
        return self._read_species_output("meanSize", species)

    def mean_trophic_level(self, species: str | None = None) -> pd.DataFrame:
        """Read mean trophic level time series."""
        return self._read_species_output("meanTL", species)

    def spatial_biomass(self, filename: str) -> xr.Dataset:
        """Read spatial (gridded) biomass output from NetCDF."""
        return self.read_netcdf(filename)

    # --- 2D output convenience methods (ByAge, BySize, ByTL) ---

    def biomass_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read biomass broken down by age class."""
        return self._read_2d_output("biomassByAge", species)

    def biomass_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read biomass broken down by size class."""
        return self._read_2d_output("biomassBySize", species)

    def biomass_by_tl(self, species: str | None = None) -> pd.DataFrame:
        """Read biomass broken down by trophic level."""
        return self._read_2d_output("biomassByTL", species)

    def abundance_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read abundance broken down by age class."""
        return self._read_2d_output("abundanceByAge", species)

    def abundance_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read abundance broken down by size class."""
        return self._read_2d_output("abundanceBySize", species)

    def abundance_by_tl(self, species: str | None = None) -> pd.DataFrame:
        """Read abundance broken down by trophic level."""
        return self._read_2d_output("abundanceByTL", species)

    def yield_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read yield biomass broken down by age class."""
        return self._read_2d_output("yieldByAge", species)

    def yield_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read yield biomass broken down by size class."""
        return self._read_2d_output("yieldBySize", species)

    def yield_n_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read yield abundance broken down by age class."""
        return self._read_2d_output("yieldNByAge", species)

    def yield_n_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read yield abundance broken down by size class."""
        return self._read_2d_output("yieldNBySize", species)

    def diet_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read diet composition broken down by age class."""
        return self._read_2d_output("dietByAge", species)

    def diet_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read diet composition broken down by size class."""
        return self._read_2d_output("dietBySize", species)

    def mean_size_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read mean size broken down by age class."""
        return self._read_2d_output("meanSizeByAge", species)

    def mean_tl_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read mean trophic level broken down by size class."""
        return self._read_2d_output("meanTLBySize", species)

    def mean_tl_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read mean trophic level broken down by age class."""
        return self._read_2d_output("meanTLByAge", species)

    # --- Additional 1D output methods ---

    def yield_abundance(self, species: str | None = None) -> pd.DataFrame:
        """Read yield in abundance (number of individuals caught)."""
        return self._read_species_output("yieldN", species)

    def mortality_rate(self, species: str | None = None) -> pd.DataFrame:
        """Read mortality rate time series."""
        return self._read_species_output("mortalityRate", species)

    def fishery_yield(self, species: str | None = None) -> pd.DataFrame:
        """Read fishery-specific yield (biomass)."""
        return self._read_species_output("fisheryYieldBiomass", species)

    def fishery_yield_by_age(self, species: str | None = None) -> pd.DataFrame:
        """Read fishery yield by age class."""
        return self._read_2d_output("fisheryYieldByAge", species)

    def fishery_yield_by_size(self, species: str | None = None) -> pd.DataFrame:
        """Read fishery yield by size class."""
        return self._read_2d_output("fisheryYieldBySize", species)

    def bioen_ingestion(self, species: str | None = None) -> pd.DataFrame:
        """Read bioenergetics ingestion rate."""
        return self._read_species_output("ingestion", species)

    def bioen_maintenance(self, species: str | None = None) -> pd.DataFrame:
        """Read bioenergetics maintenance cost."""
        return self._read_species_output("maintenance", species)

    def bioen_net_energy(self, species: str | None = None) -> pd.DataFrame:
        """Read bioenergetics net energy."""
        return self._read_species_output("meanEnet", species)

    # --- Special outputs ---

    def size_spectrum(self) -> pd.DataFrame:
        """Read size spectrum output (no species column).

        Concatenates all osm_sizeSpectrum*.csv files.
        """
        if self._in_memory:
            raise FileNotFoundError(
                "In-memory OsmoseResults does not support size_spectrum output. "
                "This output family is not captured by the build helpers used by "
                "from_outputs(); use a disk-backed OsmoseResults if you need it."
            )
        pattern = f"{self.prefix}_sizeSpectrum*.csv"
        frames = []
        for filepath in _find_output_files(self.output_dir, pattern):
            df = _read_output_csv(filepath)
            frames.append(df)
        if not frames:
            self._raise_if_strict(pattern)
            _log.info("No files matching '%s' in %s", pattern, self.output_dir)
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # --- Spatial output methods ---

    def spatial_abundance(self, filename: str) -> xr.Dataset:
        """Read spatial (gridded) abundance output from NetCDF."""
        return self.read_netcdf(filename)

    def spatial_size(self, filename: str) -> xr.Dataset:
        """Read spatial (gridded) size output from NetCDF."""
        return self.read_netcdf(filename)

    def spatial_yield(self, filename: str) -> xr.Dataset:
        """Read spatial (gridded) yield output from NetCDF."""
        return self.read_netcdf(filename)

    def spatial_ltl(self, filename: str) -> xr.Dataset:
        """Read spatial (gridded) lower trophic level output from NetCDF."""
        return self.read_netcdf(filename)

    # --- Internal helpers ---

    def _read_2d_output(self, output_type: str, species: str | None = None) -> pd.DataFrame:
        """Read 2D CSV output files and melt to long format.

        Files are expected to match: {prefix}_{output_type}*.csv
        Wide columns (bins) are melted into rows.
        Returns DataFrame with columns: time, species, bin, value.
        """
        if self._in_memory:
            if output_type not in self._csv_cache:
                raise FileNotFoundError(
                    f"In-memory OsmoseResults has no '{output_type}' output. "
                    f"Available: {sorted(self._csv_cache.keys())}"
                )
            return _melt_wide_to_long(self._csv_cache[output_type], species)

        pattern = f"{self.prefix}_{output_type}*.csv"
        frames = []
        for filepath in _find_output_files(self.output_dir, pattern):
            if not _matches_output_type(filepath.stem, output_type, self.prefix):
                continue
            df = _read_output_csv(filepath)
            sp_name = _extract_species(filepath.stem, output_type, self.prefix)
            if sp_name is None:
                sp_name = "all"
            time_col = df.columns[0]
            bin_cols = list(df.columns[1:])

            melted = df.melt(
                id_vars=[time_col],
                value_vars=bin_cols,
                var_name="bin",
                value_name="value",
            )
            melted = melted.rename(columns={time_col: "time"})  # type: ignore[assignment]
            melted["species"] = sp_name
            melted = melted[["time", "species", "bin", "value"]]  # type: ignore[assignment]
            frames.append(melted)

        if not frames:
            self._raise_if_strict(pattern)
            _log.info("No files matching '%s' in %s", pattern, self.output_dir)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if species:
            combined = combined[combined["species"] == species]  # type: ignore[assignment]
        return combined  # type: ignore[return-value]

    def _read_species_output(self, output_type: str, species: str | None) -> pd.DataFrame:
        """Read CSV output files for a given output type.

        Files are expected to match: {prefix}_{output_type}*.csv
        Each file's data gets a 'species' column derived from the filename.
        Results are cached by output_type for repeated reads.
        """
        if self._in_memory:
            if output_type not in self._csv_cache:
                raise FileNotFoundError(
                    f"In-memory OsmoseResults has no '{output_type}' output. "
                    f"Available: {sorted(self._csv_cache.keys())}"
                )
            combined = self._csv_cache[output_type]
            if species:
                combined = combined[combined["species"] == species]
            return combined

        cache_key = output_type
        if cache_key not in self._csv_cache:
            pattern = f"{self.prefix}_{output_type}*.csv"
            frames = []
            for filepath in _find_output_files(self.output_dir, pattern):
                if not _matches_output_type(filepath.stem, output_type, self.prefix):
                    continue
                df = _read_output_csv(filepath)
                sp_name = _extract_species(filepath.stem, output_type, self.prefix)
                if sp_name is None:
                    sp_name = "all"
                df["species"] = sp_name
                frames.append(df)

            if not frames:
                self._raise_if_strict(pattern)
                _log.info("No files matching '%s' in %s", pattern, self.output_dir)
                return pd.DataFrame()

            self._csv_cache[cache_key] = pd.concat(frames, ignore_index=True)

        combined = self._csv_cache[cache_key]
        if species:
            combined = combined[combined["species"] == species]  # type: ignore[assignment]
        return combined  # type: ignore[return-value]

    # Type-to-method mapping for export_dataframe
    _EXPORT_MAP: dict[str, tuple[str, str]] = {
        # 1D types: (internal_output_type, method_type)
        "biomass": ("biomass", "1d"),
        "abundance": ("abundance", "1d"),
        "yield": ("yield", "1d"),
        "mortality": ("mortalityRate", "1d"),
        "trophic": ("meanTL", "1d"),
        "yield_n": ("yieldN", "1d"),
        "mortality_rate": ("mortalityRate", "1d"),
        "fishery_yield": ("fisheryYieldBiomass", "1d"),
        "bioen_ingestion": ("ingestion", "1d"),
        "bioen_maintenance": ("maintenance", "1d"),
        "bioen_net_energy": ("meanEnet", "1d"),
        # 2D types
        "biomass_by_age": ("biomassByAge", "2d"),
        "biomass_by_size": ("biomassBySize", "2d"),
        "biomass_by_tl": ("biomassByTL", "2d"),
        "abundance_by_age": ("abundanceByAge", "2d"),
        "abundance_by_size": ("abundanceBySize", "2d"),
        "abundance_by_tl": ("abundanceByTL", "2d"),
        "yield_by_age": ("yieldByAge", "2d"),
        "yield_by_size": ("yieldBySize", "2d"),
        "yield_n_by_age": ("yieldNByAge", "2d"),
        "yield_n_by_size": ("yieldNBySize", "2d"),
        "mean_size_by_age": ("meanSizeByAge", "2d"),
        "mean_tl_by_age": ("meanTLByAge", "2d"),
        "mean_tl_by_size": ("meanTLBySize", "2d"),
        "fishery_yield_by_age": ("fisheryYieldByAge", "2d"),
        "fishery_yield_by_size": ("fisheryYieldBySize", "2d"),
        # Special types
        "diet": ("dietMatrix", "special_diet"),
        "size_spectrum": ("sizeSpectrum", "special_spectrum"),
    }

    def export_dataframe(self, output_type: str, species: str | None = None) -> pd.DataFrame:
        """Return the DataFrame for any supported output type.

        Args:
            output_type: One of the keys from the Results page dropdown
                (e.g., 'biomass', 'biomass_by_age', 'diet', 'size_spectrum').
            species: Optional species filter. Ignored for size_spectrum.

        Returns:
            DataFrame with the requested data, or empty DataFrame if unknown type.
        """
        entry = self._EXPORT_MAP.get(output_type)
        if entry is None:
            _log.warning("Unknown output type for export_dataframe: %r", output_type)
            return pd.DataFrame()

        internal_type, method_type = entry

        if method_type == "1d":
            return self._read_species_output(internal_type, species)
        elif method_type == "2d":
            return self._read_2d_output(internal_type, species)
        elif method_type == "special_diet":
            return self._read_species_output(internal_type, species)
        elif method_type == "special_spectrum":
            return self.size_spectrum()

        return pd.DataFrame()

    def close_cache(self) -> None:
        """Close all open NetCDF datasets in the cache and clear all caches."""
        for ds in self._nc_cache.values():
            ds.close()
        self._nc_cache.clear()
        self._csv_cache.clear()

    def close(self) -> None:
        """Close any cached NetCDF datasets."""
        self.close_cache()
