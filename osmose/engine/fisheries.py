"""Per-fishery fishing (OSMOSE v4 ``fisheries.*``), Java's model (spec 2026-11-24).

Java (``FishingGear``/``FisheryMapSet``) fishes each school with the SUM over fisheries ``f`` of
``q[sp, f] * rate_f * sel_f(school) * map_f(cell, step)``. ``config._parse_fisheries`` instead
reduces fisheries to one set of per-species arrays -- exact only when a config is
*species-reducible* (each species in at most one fishery, catchability 1, one shared map with no
schedule gap). :func:`build_fishery_model` returns a :class:`FisheryModel` ONLY for configs that are
not reducible; everything else keeps the per-species path (and its bit-identical results).

Each fishery's term reuses the per-species formula of ``mortality._precompute_effective_rates``
with that fishery's own rate, selectivity, map and seasonality, so a reducible config forced
through this path gives the same rates (tests/test_fisheries_per_fishery.py).

Maps (``FisheryMapSet``): ``fisheries.movement.fishery.mapK`` names the fishery map ``K`` belongs
to, ``fisheries.movement.file.mapK`` its CSV, and it applies on ``.steps.mapK`` (default all steps
of the year) of ``.years.mapK`` or ``.initialyear/.lastyear.mapK`` (default all years); a later
map overrides an earlier one on a shared step. A step no map covers, a map without a file, and a
negative/NaN cell all fish nothing, as in Java. Deliberate divergence: a fishery with NO map keys
at all fishes everywhere (factor 1, the per-species path's behaviour) instead of Java's
"never scheduled -> never fishes". Maps are used raw; Java's mean-1 normalisation (G5) is a
separate, owner-approved follow-up.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NO_MAP = -2  # fishery defines no maps at all: factor 1 (see module docstring)
OFF = -1  # step covered by no map of this fishery: no fishing


@dataclass(frozen=True)
class FisheryModel:
    names: list[str]
    rate: NDArray[np.float64]  # (F,) base rate per year
    sel_type: NDArray[np.int32]  # (F,) 0 age knife-edge, 1 logistic, else length knife-edge
    sel_a50: NDArray[np.float64]  # (F,)
    sel_l50: NDArray[np.float64]  # (F,)
    sel_slope: NDArray[np.float64]  # (F,)
    catchability: NDArray[np.float64]  # (n_species, F)
    discard: NDArray[np.float64]  # (n_species, F)
    season: NDArray[np.float64] | None  # (F, ndt) weights summing to 1, or None (uniform)
    maps: list[NDArray[np.float64] | None]  # raw maps (row-flipped like every spatial CSV)
    map_files: list[str | None]  # config value of each map's file (None = no file)
    map_index: NDArray[np.int32]  # (F, n_steps): index into maps, OFF, or NO_MAP

    @property
    def n_fisheries(self) -> int:
        return len(self.names)


def _clean_name(name: str) -> str:
    # Java FishingGear.init: name.replaceAll("_", "").replaceAll("-", "")
    return name.strip().replace("_", "").replace("-", "").lower()


def _matrix(path: Path, species_names: list[str], n_fisheries: int) -> NDArray[np.float64]:
    """Species x fishery matrix (rows by species name, columns in fishery order)."""
    df = pd.read_csv(path, index_col=0)
    out = np.zeros((len(species_names), n_fisheries), dtype=np.float64)
    rows = {str(r).strip().lower(): i for i, r in enumerate(df.index)}
    for sp, name in enumerate(species_names):
        r = rows.get(name.strip().lower())
        if r is None:
            continue
        vals = df.iloc[r].to_numpy(dtype=np.float64)[:n_fisheries]
        out[sp, : len(vals)] = vals
    return out


def _ints(raw: str) -> list[int]:
    return [int(float(v)) for v in str(raw).replace(",", ";").split(";") if v.strip()]


def _season_row(cfg: dict[str, str], f: int, ndt: int, require_file) -> NDArray[np.float64] | None:
    vals: NDArray[np.float64] | None = None
    file_key = cfg.get(f"fisheries.seasonality.file.fsh{f}", "")
    if file_key:
        path = require_file(file_key, f"fisheries.seasonality.file.fsh{f}")
        vals = pd.read_csv(path, sep=";").iloc[:, 1].to_numpy(dtype=np.float64)
    else:
        inline = cfg.get(f"fisheries.seasonality.fsh{f}", "")
        if inline:
            try:
                vals = np.array(
                    [float(v) for v in inline.split(";") if v.strip()], dtype=np.float64
                )
            except ValueError as exc:
                warnings.warn(
                    f"Invalid fisheries.seasonality.fsh{f}: {inline!r} ({exc})", stacklevel=3
                )
    if vals is None or len(vals) < ndt or vals[:ndt].sum() <= 0:
        return None
    return vals[:ndt] / vals[:ndt].sum()


def build_fishery_model(
    cfg: dict[str, str],
    species_names: list[str],
    n_dt_per_year: int,
    n_steps: int,
    require_file,
    load_spatial_csv,
    reduce: bool = True,
) -> FisheryModel | None:
    """Per-fishery tables for a v4 fisheries config, or None if it is species-reducible.

    ``require_file(value, key) -> Path`` and ``load_spatial_csv(path) -> array`` are injected from
    ``engine.config`` so this module shares its path resolution and CSV orientation.
    ``reduce=False`` returns the model even for a reducible config (the equivalence test forces a
    reducible config through the per-fishery path this way).
    """
    n_f = int(cfg.get("simulation.nfisheries", "0") or 0)
    if n_f <= 0 or not cfg.get("fisheries.catchability.file"):
        return None
    n_sp = len(species_names)
    names = [cfg.get(f"fisheries.name.fsh{f}", f"fishery{f}") for f in range(n_f)]
    q = _matrix(
        require_file(cfg["fisheries.catchability.file"], "fisheries.catchability.file"),
        species_names,
        n_f,
    )
    discard = (
        _matrix(
            require_file(cfg["fisheries.discards.file"], "fisheries.discards.file"),
            species_names,
            n_f,
        )
        if cfg.get("fisheries.discards.file")
        else np.zeros((n_sp, n_f))
    )
    rate = np.array([float(cfg.get(f"fisheries.rate.base.fsh{f}", "0") or 0) for f in range(n_f)])
    sel_type = np.array(
        [int(float(cfg.get(f"fisheries.selectivity.type.fsh{f}", "0") or 0)) for f in range(n_f)],
        dtype=np.int32,
    )
    a50 = np.array(
        [float(cfg.get(f"fisheries.selectivity.a50.fsh{f}", "0") or 0) for f in range(n_f)]
    )
    l50 = np.array(
        [float(cfg.get(f"fisheries.selectivity.l50.fsh{f}", "0") or 0) for f in range(n_f)]
    )
    slope = np.array(
        [float(cfg.get(f"fisheries.selectivity.slope.fsh{f}", "1") or 1) for f in range(n_f)]
    )
    rows = [_season_row(cfg, f, n_dt_per_year, require_file) for f in range(n_f)]
    season = None
    if any(r is not None for r in rows):
        season = np.vstack(
            [r if r is not None else np.full(n_dt_per_year, 1.0 / n_dt_per_year) for r in rows]
        )

    # ---- maps and their schedule (FisheryMapSet.loadMaps) ----
    by_name = {_clean_name(n): f for f, n in enumerate(names)}
    map_ids = sorted(
        int(k[len("fisheries.movement.fishery.map") :])
        for k in cfg
        if k.startswith("fisheries.movement.fishery.map")
        and k[len("fisheries.movement.fishery.map") :].isdigit()
    )
    n_year = max(1, int(np.ceil(n_steps / n_dt_per_year)))
    map_index = np.full((n_f, n_steps), NO_MAP, dtype=np.int32)
    maps: list[NDArray[np.float64] | None] = []
    files: list[str | None] = []
    for k in map_ids:
        fname = _clean_name(cfg[f"fisheries.movement.fishery.map{k}"])
        if fname not in by_name:
            raise ValueError(
                f"fisheries.movement.fishery.map{k} = {cfg[f'fisheries.movement.fishery.map{k}']!r} "
                f"names no fishery (fisheries.name.fsh*: {names})"
            )
        f = by_name[fname]
        if (map_index[f] == NO_MAP).all():
            map_index[f] = OFF  # the fishery has maps: unscheduled steps fish nothing
        steps_raw = cfg.get(f"fisheries.movement.steps.map{k}", "")
        steps = _ints(steps_raw) if steps_raw else list(range(n_dt_per_year))
        if cfg.get(f"fisheries.movement.years.map{k}", ""):
            years = _ints(cfg[f"fisheries.movement.years.map{k}"])
        else:
            y0 = int(float(cfg.get(f"fisheries.movement.initialyear.map{k}", "0") or 0))
            y1 = int(float(cfg.get(f"fisheries.movement.lastyear.map{k}", str(n_year - 1))))
            years = list(range(y0, y1 + 1))
        file_val = cfg.get(f"fisheries.movement.file.map{k}", "")
        if file_val in files:
            idx = files.index(file_val)  # Java eliminateTwinMap: same file, one map
        else:
            idx = len(maps)
            files.append(file_val or None)
            maps.append(
                load_spatial_csv(require_file(file_val, f"fisheries.movement.file.map{k}"))
                if file_val
                else None
            )
        for y in years:
            if not 0 <= y < n_year:
                continue
            for s in steps:
                if 0 <= s < n_dt_per_year and y * n_dt_per_year + s < n_steps:
                    map_index[f, y * n_dt_per_year + s] = idx

    model = FisheryModel(
        names=names,
        rate=rate,
        sel_type=sel_type,
        sel_a50=a50,
        sel_l50=l50,
        sel_slope=slope,
        catchability=q,
        discard=discard,
        season=season,
        maps=maps,
        map_files=files,
        map_index=map_index,
    )
    return None if reduce and is_species_reducible(model, cfg) else model


def is_species_reducible(model: FisheryModel, cfg: dict[str, str]) -> bool:
    """True when the per-species path (config._parse_fisheries & co.) is exactly this model.

    That path gives each species the FIRST fishery with q > 0, treats q as a flag, gives a
    fishery's seasonality to one species, and applies ``fisheries.movement.file.map0`` to every
    species at every step. So the model reduces iff, over the fisheries actually used:
    """
    caught = model.catchability > 0
    if (caught.sum(axis=1) > 1).any():
        return False  # G1: a species in more than one fishery
    if not np.all(model.catchability[caught] == 1.0):
        return False  # G2: catchability other than 1
    used = sorted(set(np.nonzero(caught)[1].tolist()))
    for f in used:
        has_season = cfg.get(f"fisheries.seasonality.file.fsh{f}") or cfg.get(
            f"fisheries.seasonality.fsh{f}"
        )
        if has_season and caught[:, f].sum() > 1:
            return False  # the per-species loader would give this seasonality to one species
    if any(k.startswith("mortality.fishing.spatial.distrib.file.sp") for k in cfg):
        return True  # legacy per-species maps take precedence (unchanged behaviour)
    shared = cfg.get("fisheries.movement.file.map0", "") or None
    for f in used:
        row = model.map_index[f]
        if (row == NO_MAP).all():
            if shared is not None:
                return False  # per-species path would apply map0 to a fishery with no maps
            continue
        if (row == OFF).any():
            return False  # G4: a scheduled gap (the fishery is off on some steps)
        if len(set(row.tolist())) != 1:
            return False  # G3: the fishery changes map over time
        if model.map_files[int(row[0])] != shared:
            return False  # G3: its map is not the shared map0 the per-species path applies
    return True


def per_fishery_rates(
    model: FisheryModel,
    species_id: NDArray[np.int32],
    age_dt: NDArray[np.int32],
    length: NDArray[np.float64],
    cell_y: NDArray[np.int32],
    cell_x: NDArray[np.int32],
    step: int,
    n_dt_per_year: int,
    n_subdt: int,
    lookup,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(fishing rate per school per sub-step, discard fraction per school) before MPA/fleet/exempt.

    Each fishery term is the per-species formula with the fishery's parameters times its
    catchability; the discard fraction is the rate-weighted mean of the fisheries' discard rates
    (deaths split across fisheries in proportion to their rates).
    """
    n = len(species_id)
    n_sp = model.catchability.shape[0]
    focal = species_id < n_sp
    sp = np.where(focal, species_id, 0)
    total = np.zeros(n, dtype=np.float64)
    disc = np.zeros(n, dtype=np.float64)
    denom = n_dt_per_year * n_subdt
    step_in_year = step % n_dt_per_year
    t = min(step, model.map_index.shape[1] - 1)
    every = np.ones(n, dtype=bool)
    for f in range(model.n_fisheries):
        q = np.where(focal, model.catchability[sp, f], 0.0)
        if not (q > 0).any() or model.rate[f] <= 0:
            continue
        mi = int(model.map_index[f, t])
        if mi == OFF:
            continue
        if model.sel_type[f] == 0:
            sel = np.where(age_dt / n_dt_per_year < model.sel_a50[f], 0.0, 1.0)
        elif model.sel_type[f] == 1:
            sel = 1.0 / (1.0 + np.exp(-model.sel_slope[f] * (length - model.sel_l50[f])))
        else:
            l50 = model.sel_l50[f]
            sel = np.where((l50 > 0) & (length < l50), 0.0, 1.0)
        if mi == NO_MAP:
            spatial = np.ones(n, dtype=np.float64)
        elif model.maps[mi] is None:
            continue  # a map without a file: Java fishes nothing
        else:
            spatial = lookup(model.maps[mi], cell_y, cell_x, every)
        base = q * model.rate[f] * sel * spatial
        if model.season is not None:
            term = base * model.season[f, step_in_year] / n_subdt
        else:
            term = base / denom
        total += term
        disc += term * model.discard[sp, f]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, disc / total, 0.0)
    return total, frac
