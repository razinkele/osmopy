"""Stage a background-species (``simulation.nbackground > 0``) config to run on the Java 4.4.1 jar.

OSMOSE 4.4.1 can run a config with background predators (Baltic: GreySeal, Cormorant), but only with
keys the OSMOPY configs don't ship: an inline ``species.biomass.spN`` time-series, the background
species in the predation-accessibility + catchability/discards matrices, explicit movement maps,
``simulation.nschool.spN``, ``output.diet.stage.threshold.spN``, and ``output.cutoff.enabled=false``
(a 4.4.1 ``OutputRegion.include`` OOB with nbackground>0). This module materialises all of that on a
STAGED copy of the config (never ``data/``) so the UI Java path can run such configs.

Scope is **Baltic-specific**: the accessibility + diet-stage tables are hand-authored for
GreySeal/Cormorant (validated in sub-project A). ``background_staging_supported`` gates other configs.

This module must NOT import ``osmose.runner`` or ``ui.pages.run`` (``write_temp_config``) — the
orchestrator takes an ALREADY-STAGED dir; the caller stages first. (Else: runner -> this -> ui.run ->
runner import cycle.)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

# Authored accessibility (prey -> value) for each background PREDATOR, from baltic_param-background.csv
# diet intent + size-ratio windows (validated in sub-project A).
# `cod` is the PRE-SPLIT prey row, still used by `data/baltic_ev` and `data/baltic-fine`.
# Production `data/baltic` split it into `cod_west`/`cod_east` (see the cod E/W disaggregation), and
# because `augment_accessibility` looks prey up by row name with a `0.0` default, the stale `cod`
# key silently zeroed seal AND cormorant access to BOTH cod stocks on every Java arm. All three
# names are carried so the table serves split and unsplit configs alike; `_warn_unmatched_prey`
# makes any future rename loud instead of silent.
BG_ACCESS: dict[str, dict[str, float]] = {
    "GreySeal": {
        "herring": 0.4,
        "sprat": 0.4,
        "cod": 0.3,
        "cod_west": 0.3,
        "cod_east": 0.3,
        "flounder": 0.2,
        "pikeperch": 0.1,
    },
    "Cormorant": {
        "perch": 0.4,
        "herring": 0.3,
        "sprat": 0.3,
        "cod": 0.1,
        "cod_west": 0.1,
        "cod_east": 0.1,
    },
}

# Per-predator diet-stage size threshold (cm): juvenile (<thr) / adult (>=thr). 4.4.1 requires
# output.diet.stage.threshold.spN for every species in the predation-accessibility universe.
BG_DIET_STAGE_THRESHOLD: dict[str, int] = {"GreySeal": 90, "Cormorant": 65}


def inline_biomass_series(nc_path: str | Path, varname: str) -> list[float]:
    """Per-step domain-total biomass (length = n time steps) for a background species' NetCDF var."""
    ds = xr.open_dataset(nc_path)
    a = ds[varname]
    spatial = [d for d in a.dims if d != a.dims[0]]  # sum all but the leading (time) dim
    return [float(v) for v in np.nan_to_num(a.sum(dim=spatial).values)]


def _warn_unmatched_prey(predators: dict[str, dict[str, float]], prey_rows: set[str]) -> None:
    """Loudly flag authored prey keys that match no row — the silent-`0.0` trap.

    `augment_accessibility` defaults an unmatched prey key to 0.0, so a renamed row (as happened
    when `cod` became `cod_west`/`cod_east`) silently removes that predator's access instead of
    failing. A predator whose keys ALL miss is a hard error; partial misses warn, because carrying
    both split and unsplit names means some keys are legitimately unused per config.
    """
    for name, access in predators.items():
        matched = {k for k in access if k in prey_rows}
        if access and not matched:
            raise ValueError(
                f"background predator {name!r}: none of its authored prey keys "
                f"{sorted(access)} match any prey row in the accessibility matrix "
                f"{sorted(prey_rows)}. Staging would silently give it 0.0 access to everything."
            )
        unmatched = sorted(set(access) - matched)
        if unmatched:
            warnings.warn(
                f"background predator {name!r}: authored prey keys {unmatched} match no prey row "
                f"and contribute nothing. Expected when a config predates the cod E/W split; "
                f"a surprise otherwise.",
                stacklevel=3,
            )


def augment_accessibility(csv_path: Path, predators: dict[str, dict[str, float]]) -> None:
    """Add background predators as columns (authored prey access) + apex prey rows (0), in place.

    Idempotent per name: a predator that already has a column is not given a second one, and a
    predator that already has a prey row is not given a second row. `header += names` unconditionally
    used to emit a duplicate `Cormorant` column on `data/baltic`, where Cormorant is already a
    column — leaving Java to pick between two columns for the same predator.
    """
    lines = [ln for ln in csv_path.read_text().splitlines() if ln.strip()]
    header = lines[0].split(";")
    existing_cols = {h.strip() for h in header}
    prey_rows = {ln.split(";")[0].strip() for ln in lines[1:]}
    _warn_unmatched_prey(predators, prey_rows)

    names = [p for p in predators if p not in existing_cols]  # only genuinely new columns
    header = header + names
    rows = [header]
    for ln in lines[1:]:
        cells = ln.split(";")
        prey = cells[0].strip()
        cells += [str(predators[p].get(prey, 0.0)) for p in names]
        rows.append(cells)
    ncol = len(header)
    for p in predators:  # apex prey rows: 0 accessibility to every predator
        if p not in prey_rows:
            rows.append([p] + ["0"] * (ncol - 1))
    csv_path.write_text("\n".join(";".join(c) for c in rows) + "\n")


def _augment_catchability_discards(stage_dir: Path, predators: dict) -> None:
    """Append zero-catchability/discards rows for each background predator to the STAGED matrices.

    4.4.1's Matrix resolves prey indices from the accessibility-matrix universe, so any row added
    there must also appear in catchability + discards (even as zeros). Staged copies only.
    """
    for fname in ("fishery-catchability.csv", "fishery-discards.csv"):
        fpath = stage_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text()
        n_fisheries = len(text.splitlines()[0].split(",")) - 1
        zeros = ",".join(["0"] * n_fisheries)
        for sp_name in predators:
            text += f"{sp_name},{zeros}\n"
        fpath.write_text(text)


def _write_background_movement_maps(
    stage: Path,
    bg_species: list[tuple[str, int]],
    all_steps: list[int],
    ref_map_csv: Path,
) -> list[str]:
    """Create uniform sea-distribution CSV maps for background species and return the config lines.

    BackgroundMapSet (4.4.1) requires explicit movement.{species,file,steps,class}.mapN entries
    for every background species and class — it has no 'random' mode, unlike focal species.
    We copy the sea-cell mask from an existing focal-species map (1=sea, -99=land) and use it
    as a uniform all-sea distribution for wide-ranging top predators.
    """
    ref_lines = ref_map_csv.read_text().splitlines()
    ref_arr = [
        [float(x) for x in ln.strip().split(";") if x.strip()] for ln in ref_lines if ln.strip()
    ]
    # Replace any positive value with 1 (uniform); keep land as -99
    sea_rows = [";".join("-99" if v < -90 else "1" for v in row) for row in ref_arr]
    sea_text = "\n".join(sea_rows) + "\n"

    maps_dir = stage / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    config_lines: list[str] = []
    step_str = ";".join(str(s) for s in all_steps)

    # Find the next free map index in the stage master to avoid collision
    master_text = (stage / "osm_all-parameters.csv").read_text()
    existing = [int(m.group(1)) for m in re.finditer(r"movement\.species\.map(\d+)", master_text)]
    next_idx = (max(existing) + 1) if existing else 26

    for sp_name, n_class in bg_species:
        map_file = maps_dir / f"background_{sp_name.lower()}_all.csv"
        map_file.write_text(sea_text)
        for cls in range(n_class):
            idx = next_idx
            next_idx += 1
            rel_path = f"maps/background_{sp_name.lower()}_all.csv"
            config_lines.append(f"movement.species.map{idx} ; {sp_name}")
            config_lines.append(f"movement.file.map{idx} ; {rel_path}")
            config_lines.append(f"movement.steps.map{idx} ; {step_str}")
            config_lines.append(f"movement.class.map{idx} ; {cls}")
    return config_lines


def _background_species(config: dict) -> list[tuple[str, str]]:
    """[(idx, name), ...] for each species.type.spN == 'background' in the config."""
    out = []
    for key, val in config.items():
        if key.startswith("species.type.sp") and str(val).strip().lower() == "background":
            idx = key[len("species.type.sp") :]
            if idx.isdigit():
                out.append((idx, str(config.get(f"species.name.sp{idx}", "")).strip()))
    return out


def background_staging_supported(config: dict) -> bool:
    """True iff every background species in *config* has a hand-authored staging table entry."""
    names = [name for _, name in _background_species(config)]
    return bool(names) and all(n in BG_ACCESS for n in names)


def stage_background_for_java(stage_dir: Path, raw_config: dict) -> dict[str, str]:
    """Materialise the 4.4.1 background-species keys/matrices into an already-staged config dir.

    *stage_dir* is the output of ``write_temp_config`` (flat ``osm_all-parameters.csv`` + copied data,
    incl. the predator NetCDF + ``maps/``). Returns the extra ``-P`` overrides the run needs (the
    cutoff workaround), which the caller passes to ``OsmoseRunner.run(overrides=...)``. Staged-copy
    only — never touches ``data/``. Requires ``background_staging_supported(raw_config)``.
    """
    stage_dir = Path(stage_dir)
    master = stage_dir / "osm_all-parameters.csv"
    ndt = int(float(raw_config.get("simulation.time.ndtperyear", "24") or "24"))
    nc_matches = sorted(stage_dir.glob("*predator*biomass*.nc"))  # the background-forcing NetCDF
    if not nc_matches:
        raise FileNotFoundError(
            f"Background staging needs a predator-biomass NetCDF (*predator*biomass*.nc) in the "
            f"staged config dir {stage_dir}, but none was found. Was the source data copied?"
        )
    nc = nc_matches[0]
    ref_maps = sorted((stage_dir / "maps").glob("*juvenile*.csv")) or sorted(
        (stage_dir / "maps").glob("*.csv")
    )
    if not ref_maps:
        raise FileNotFoundError(
            f"Background staging needs a reference sea-mask movement map (maps/*.csv) in {stage_dir}, "
            "but none was found."
        )
    ref_map = ref_maps[0]

    extra: list[str] = []
    predators: dict[str, dict[str, float]] = {}
    bg_for_maps: list[tuple[str, int]] = []
    for idx, name in _background_species(raw_config):
        nclass = int(float(raw_config.get(f"species.nclass.sp{idx}", "1") or "1"))
        series = inline_biomass_series(nc, name)
        extra.append(f"species.biomass.sp{idx} ; " + ";".join(f"{v:.6g}" for v in series))
        extra.append(f"species.biomass.nsteps.year.sp{idx} ; {ndt}")
        extra.append(f"simulation.nschool.sp{idx} ; 10")
        extra.append(f"output.diet.stage.threshold.sp{idx} ; {BG_DIET_STAGE_THRESHOLD[name]}")
        bg_for_maps.append((name, nclass))
        predators[name] = BG_ACCESS[name]
    # ONE movement-map call for ALL background species — the helper computes the next free map
    # index from the master (not updated until after this), so per-species calls would collide.
    extra.extend(_write_background_movement_maps(stage_dir, bg_for_maps, list(range(ndt)), ref_map))
    extra.append(
        "output.cutoff.enabled ; false"
    )  # belt-and-suspenders; the -P override is authoritative
    master.write_text(master.read_text() + "\n".join(extra) + "\n")
    augment_accessibility(stage_dir / "predation-accessibility.csv", predators)
    _augment_catchability_discards(stage_dir, predators)
    return {"output.cutoff.enabled": "false"}
