"""Read OSMOSE simulation output files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


class OsmoseResults:
    """Read and query OSMOSE simulation outputs.

    OSMOSE writes output as CSV and/or NetCDF files. This class provides
    a unified interface to access biomass, abundance, yield, diet, and
    mortality data.
    """

    def __init__(self, output_dir: Path, prefix: str = "osm"):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self._nc_cache: dict[str, xr.Dataset] = {}

    def list_outputs(self) -> list[str]:
        """List all output files in the output directory."""
        files = []
        for f in sorted(self.output_dir.iterdir()):
            if f.suffix in (".csv", ".nc"):
                files.append(f.name)
        return files

    def read_csv(self, pattern: str) -> dict[str, pd.DataFrame]:
        """Read CSV output files matching a glob pattern.

        Returns dict mapping filename to DataFrame.
        """
        result = {}
        for f in sorted(self.output_dir.glob(pattern)):
            result[f.stem] = pd.read_csv(f)
        return result

    def read_netcdf(self, filename: str) -> xr.Dataset:
        """Read a NetCDF output file, with caching."""
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
        return self._read_species_output("mortality", species)

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

    # --- Special outputs ---

    def size_spectrum(self) -> pd.DataFrame:
        """Read size spectrum output (no species column).

        Concatenates all osm_sizeSpectrum*.csv files.
        """
        pattern = f"{self.prefix}_sizeSpectrum*.csv"
        frames = []
        for filepath in sorted(self.output_dir.glob(pattern)):
            df = pd.read_csv(filepath)
            frames.append(df)
        if not frames:
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
        pattern = f"{self.prefix}_{output_type}*.csv"
        frames = []
        for filepath in sorted(self.output_dir.glob(pattern)):
            df = pd.read_csv(filepath)
            # Extract species name from filename
            parts = filepath.stem.split("_", 2)
            sp_name = parts[2] if len(parts) > 2 else filepath.stem

            # First column is time; remaining columns are bins
            time_col = df.columns[0]
            bin_cols = list(df.columns[1:])

            melted = df.melt(
                id_vars=[time_col],
                value_vars=bin_cols,
                var_name="bin",
                value_name="value",
            )
            melted = melted.rename(columns={time_col: "time"})
            melted["species"] = sp_name
            melted = melted[["time", "species", "bin", "value"]]
            frames.append(melted)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if species:
            combined = combined[combined["species"] == species]
        return combined

    def _read_species_output(self, output_type: str, species: str | None) -> pd.DataFrame:
        """Read CSV output files for a given output type.

        Files are expected to match: {prefix}_{output_type}*.csv
        Each file's data gets a 'species' column derived from the filename.
        """
        pattern = f"{self.prefix}_{output_type}*.csv"
        frames = []
        for filepath in sorted(self.output_dir.glob(pattern)):
            df = pd.read_csv(filepath)
            # Extract species name from filename (e.g., osm_biomass_Anchovy.csv -> Anchovy)
            parts = filepath.stem.split("_", 2)
            sp_name = parts[2] if len(parts) > 2 else filepath.stem
            df["species"] = sp_name
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if species:
            combined = combined[combined["species"] == species]
        return combined

    def close(self) -> None:
        """Close any cached NetCDF datasets."""
        for ds in self._nc_cache.values():
            ds.close()
        self._nc_cache.clear()
