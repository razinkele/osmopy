"""OSMOSE 4.4.x output regions (``Configuration.java``, ``OutputRegion``, ``FisheryMapSet``).

Region 0 is the whole domain: the files every run writes, unchanged. A region k > 0 is declared by
``output.regions.<x>.rg<k>`` keys: ``output.regions.name.rg<k>`` names it (``_`` and ``-`` removed;
what remains must be alphanumeric) and ``output.regions.enabled.rg<k>`` can turn it off (it is on
unless that key is present and not ``true``, as Java's ``!canFind || getBoolean`` reads it). Its
extent is a series of maps, as for a fishery: ``output.regions.movement.region.map<m>`` names the
region a map belongs to, ``output.regions.movement.file.map<m>`` is a CSV map (or ``null``), and
``.steps.map<m>`` (steps of a year, default all) and ``.years.map<m>`` or
``.initialYear.map<m>`` / ``.lastYear.map<m>`` (inclusive, default the whole run) say when it
applies. A school is in a region at a step when it is in the domain and the region's map at that
step is above 0 at its cell; a step without a map means outside.

Per region the engine writes the families Java writes per region and this engine writes at all:
biomass, abundance, mortality, yield, yield in numbers, the age and size distributions, mean size
and mean trophic level, as ``<prefix>_<output>-<region>_Simu0.csv`` (mortality, per species:
``Mortality/<prefix>_mortalityRate-<region>-<species>_Simu0.csv``, as ``MortalityOutput`` names
it). With ``output.step0.include`` a region's initial record is taken after the first movement, as
Java records it. The predator-pressure and
diet outputs are not written per region: they are accumulated by the predation kernel over all
predators, not per school.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from osmose.engine.movement_maps import _load_csv_grid, _resolve_path
from osmose.engine.state import SchoolState

_RG = re.compile(r"^output\.regions\.[a-z0-9_.]+\.rg(\d+)$")
PREFIX = "output.regions.movement"


@dataclass
class OutputRegion:
    index: int
    name: str
    index_maps: NDArray[np.int32]  # map index per simulation step, -1 = outside
    maps: list[NDArray[np.float64] | None]

    def mask(self, state: SchoolState, step: int) -> NDArray[np.bool_]:
        """Schools in the region at ``step`` (Java ``OutputRegion.contains``)."""
        out = np.zeros(len(state), dtype=bool)
        if step < 0 or step >= len(self.index_maps) or not len(state):
            return out
        k = int(self.index_maps[step])
        grid = self.maps[k] if k >= 0 else None
        if grid is None:
            return out
        located = ~state.is_out & (state.cell_x >= 0) & (state.cell_y >= 0)
        ys, xs = state.cell_y[located], state.cell_x[located]
        out[located] = grid[ys, xs] > 0
        return out


def _set(cfg: dict[str, str], key: str) -> bool:
    value = str(cfg.get(key, "")).strip()
    return bool(value) and value.lower() != "null"


def region_indices(cfg: dict[str, str]) -> list[int]:
    """Enabled output regions other than the whole domain, in index order."""
    found = sorted({int(m.group(1)) for k in cfg if (m := _RG.match(k))} - {0})
    out = []
    for k in found:
        key = f"output.regions.enabled.rg{k}"
        if key not in cfg or str(cfg[key]).strip().lower() == "true":
            out.append(k)
    return out


def _ints(value: str) -> list[int]:
    return [int(float(v)) for v in re.split(r"[;,]", value) if v.strip()]


def parse_output_regions(
    cfg: dict[str, str], n_dt_per_year: int, n_steps: int, ny: int, nx: int
) -> list[OutputRegion]:
    """The output regions of a config; empty when it declares none."""
    if (
        "output.region.enabled.rg0" in cfg
        and str(cfg["output.region.enabled.rg0"]).strip().lower() != "true"
    ):
        raise ValueError(
            "output.region.enabled.rg0 is not true: Java then writes no whole-domain files, which "
            "this engine always writes; remove the key"
        )
    if len([k for k in cfg if re.match(r"^output\.region\.[a-z0-9_.]+\.rg0$", k)]) > 1:
        raise ValueError("Output region 0 is the whole grid and cannot be redefined (as in Java)")
    config_dir = cfg.get("_osmose.config.dir", "")
    n_year = -(-n_steps // n_dt_per_year)
    regions = []
    for k in region_indices(cfg):
        raw_name = str(cfg.get(f"output.regions.name.rg{k}", "")).strip()
        name = raw_name.replace("_", "").replace("-", "")
        if not name or not re.fullmatch(r"[A-Za-z0-9]+", name):
            raise ValueError(
                f"output.regions.name.rg{k} = {raw_name!r}: an output region name must be "
                "alphanumeric (underscores and dashes are removed), as Java requires"
            )
        index_maps = np.full(max(n_steps, n_dt_per_year), -1, dtype=np.int32)
        maps: list[NDArray[np.float64] | None] = []
        numbers = sorted(
            int(key.rsplit("map", 1)[1])
            for key, value in cfg.items()
            if key.startswith(f"{PREFIX}.region.map") and str(value).strip() == raw_name
        )
        for m in numbers:
            steps_key, years_key = f"{PREFIX}.steps.map{m}", f"{PREFIX}.years.map{m}"
            steps = _ints(cfg[steps_key]) if _set(cfg, steps_key) else list(range(n_dt_per_year))
            steps = [s for s in steps if s < n_dt_per_year]
            if _set(cfg, years_key):
                years = _ints(cfg[years_key])
            else:
                lo_key, hi_key = f"{PREFIX}.initialyear.map{m}", f"{PREFIX}.lastyear.map{m}"
                lo = int(cfg[lo_key]) if _set(cfg, lo_key) else 0
                hi = int(cfg[hi_key]) if _set(cfg, hi_key) else n_year - 1
                years = list(range(lo, hi + 1))
            years = [y for y in years if y < n_year]
            file_key = f"{PREFIX}.file.map{m}"
            grid = (
                _load_csv_grid(_resolve_path(cfg[file_key], config_dir), ny, nx)
                if _set(cfg, file_key)
                else None
            )
            maps.append(grid)
            for y in years:
                for s in steps:
                    t = y * n_dt_per_year + s
                    if t < len(index_maps):
                        index_maps[t] = len(maps) - 1
        regions.append(OutputRegion(k, name, index_maps, maps))
    return regions


def subset(state: SchoolState, mask: NDArray[np.bool_]) -> SchoolState:
    """The schools of ``state`` where ``mask`` is True (every field, as ``compact`` does)."""
    picked: dict[str, Any] = {}
    for f in fields(state):
        arr = getattr(state, f.name)
        picked[f.name] = None if arr is None else (arr[mask] if arr.ndim == 1 else arr[mask, :])
    return SchoolState(**picked)


def background_by_region(
    state: SchoolState, n_focal_schools: int, n_total_species: int, mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Background biomass and abundance per species, over the background schools in a region."""
    biomass = np.zeros(n_total_species, dtype=np.float64)
    abundance = np.zeros(n_total_species, dtype=np.float64)
    sel = np.zeros(len(state), dtype=bool)
    sel[n_focal_schools:] = mask[n_focal_schools:]
    if sel.any():
        np.add.at(biomass, state.species_id[sel], state.biomass[sel])
        np.add.at(abundance, state.species_id[sel], state.abundance[sel])
    return biomass, abundance


def write_region_outputs(
    region_outputs: dict[str, list[Any]], output_dir: Path, config: Any, prefix: str = "osm"
) -> None:
    """Write each region's files as Java names them: ``<prefix>_<output>-<region>_Simu0.csv``."""
    from osmose.engine import output as out

    for name, outputs in region_outputs.items():
        if not outputs:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            out._write_biomass_abundance(t, prefix, outputs, config)
            out._write_mortality_csvs(t, prefix, outputs, config)
            out._write_yield_csv(t, prefix, outputs, config)
            out._write_distribution_csvs(t, prefix, outputs, config)
            out._write_meantl_csv(t, prefix, outputs, config)
            out._write_yieldn_csv(t, prefix, outputs, config)
            out._write_meansize_csv(t, prefix, outputs, config)
            mortality = f"{prefix}_mortalityRate-"
            for f in sorted(t.rglob("*.csv")):
                rel = f.relative_to(t)
                if rel.name.startswith(mortality):
                    # MortalityOutput.init: the region comes before the species.
                    new_name = mortality + f"{name}-" + rel.name[len(mortality) :]
                else:
                    new_name = rel.name.replace("_Simu", f"-{name}_Simu", 1)
                target = output_dir / rel.parent / new_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), target)
