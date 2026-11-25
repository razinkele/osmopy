"""Java's diet stages (``SchoolStage("output.diet.stage")``) for the staged diet outputs.

Per predator index (focal species, then background species by their file index) a float32
threshold list ``output.diet.stage.threshold.sp{i}`` (none -> one stage) and a structure
(``output.diet.stage.structure.sp{i}``, else ``output.diet.stage.structure``): age (years), size
(length, cm), weight (grams: tonnes x 1e6) or tl. A school's stage is the number of leading
thresholds at or below its value, compared in float32 as Java's ``float`` getters do; resources
have one stage. Prey are staged by their value AT PREDATION (Java's ``Prey`` is a snapshot),
predators by their value when the outputs are collected (after growth and reproduction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STRUCTURES = ("age", "size", "weight", "tl")


@dataclass(frozen=True)
class DietStages:
    thresholds: list[NDArray[np.float32]]  # per runtime species (focal, then background)
    structure: NDArray[np.int32]  # index into STRUCTURES, per runtime species
    n_resources: int

    @property
    def n_stages(self) -> NDArray[np.int32]:
        return np.array([len(t) + 1 for t in self.thresholds], dtype=np.int32)

    @property
    def col_offset(self) -> NDArray[np.int32]:
        """Where each species' stage columns start on the staged prey axis; the resources take
        the last ``n_resources`` columns (one stage each)."""
        return np.concatenate([[0], np.cumsum(self.n_stages)]).astype(np.int32)

    @property
    def width(self) -> int:
        return int(self.col_offset[-1]) + self.n_resources

    def values(self, species_id, age_dt, length, weight, trophic_level, n_dt_per_year):
        """Each school's staging value, float32 (Java's float getters)."""
        sp = np.asarray(species_id)
        kind = self.structure[np.clip(sp, 0, len(self.structure) - 1)]
        v = np.where(
            kind == 0,
            np.asarray(age_dt, dtype=np.float64) / n_dt_per_year,
            np.where(
                kind == 1,
                length,
                np.where(kind == 2, np.asarray(weight) * 1e6, trophic_level),
            ),
        )
        return v.astype(np.float32)

    def stage(self, species_id, values_f32) -> NDArray[np.int32]:
        """Java ``getStage``: thresholds are increasing (checked at parse), so the count of
        leading thresholds <= value is a right-sided search."""
        out = np.zeros(len(species_id), dtype=np.int32)
        sp = np.asarray(species_id)
        for s, t in enumerate(self.thresholds):
            if len(t) == 0:
                continue
            m = sp == s
            if m.any():
                out[m] = np.searchsorted(t, values_f32[m], side="right")
        return out

    def prey_columns(self, species_id, values_f32) -> NDArray[np.int32]:
        """The staged prey column of each school (species out of range -> 0, never booked)."""
        sp = np.asarray(species_id)
        n_sp = len(self.thresholds)
        st = self.stage(sp, values_f32)
        return np.where(sp < n_sp, self.col_offset[np.clip(sp, 0, n_sp - 1)] + st, 0).astype(
            np.int32
        )


def parse_thresholds(raw: str, key: str) -> NDArray[np.float32]:
    """``10;30`` (or ``,``-separated) -> float32 thresholds, strictly increasing."""
    parts = [p for p in str(raw).replace(",", ";").split(";") if p.strip()]
    t = np.array([float(p) for p in parts], dtype=np.float32)
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError(f"{key}: diet stage thresholds must be strictly increasing, got {raw!r}")
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{key}: diet stage thresholds must be finite, got {raw!r}")
    return t


def stages_for(config) -> DietStages:
    """The run's DietStages (built once, when the resources are known)."""
    ds = getattr(config, "_diet_stages", None)
    if ds is None or ds.n_resources != len(config.resource_names):
        ds = DietStages(
            list(config.diet_stage_thresholds),
            np.asarray(config.diet_stage_structure, dtype=np.int32),
            len(config.resource_names),
        )
        object.__setattr__(config, "_diet_stages", ds)
    return ds


def _values(ds: DietStages, state, config):
    return ds.values(
        state.species_id,
        state.age_dt,
        state.length,
        state.weight,
        state.trophic_level,
        config.n_dt_per_year,
    )


def pred_rows(ds: DietStages, n_pred_species: int) -> NDArray[np.int32]:
    """Where each predator species' stage rows start (focal, then background)."""
    return np.concatenate([[0], np.cumsum(ds.n_stages[:n_pred_species])]).astype(np.int32)


def _alive(state, config) -> NDArray[np.bool_]:
    """Java ``School.isAlive``: abundance > 0 and age <= lifespan - 1 (focal schools)."""
    sp = np.clip(state.species_id, 0, len(config.lifespan_dt) - 1)
    return (state.abundance > 0) & (state.age_dt <= config.lifespan_dt[sp] - 1)


def biomass_at_step_start(state, config, resources, ocean) -> NDArray[np.float64]:
    """Java BiomassDietStageOutput.initStep: present focal schools' and every background
    school's biomass by (species, stage), then each resource's total biomass over the ocean."""
    ds = stages_for(config)
    out = np.zeros(ds.width, dtype=np.float64)
    n_sp, n_all = config.n_species, config.n_species + config.n_background
    sp = state.species_id
    focal = sp < n_sp
    keep = (focal & _alive(state, config) & ~state.is_out) | ((sp >= n_sp) & (sp < n_all))
    if keep.any():
        cols = ds.prey_columns(sp[keep], _values(ds, state, config)[keep])
        np.add.at(out, cols, state.abundance[keep] * state.weight[keep])
    if resources is not None and ds.n_resources:
        bio = np.asarray(resources.biomass, dtype=np.float64)
        mask = np.asarray(ocean, dtype=bool).ravel() if ocean is not None else None
        for r in range(ds.n_resources):
            row = bio[r] if mask is None else bio[r][mask[: bio.shape[1]]]
            out[ds.width - ds.n_resources + r] = float(np.nansum(row))
    return out


def snapshot_after_mortality(state, config, staged_rows):
    """Right after mortality, before the background schools are stripped: the background
    predators' contributions (their stage and abundance are final), and the focal schools'
    staged rows, in the order the stripped state keeps them."""
    ds = stages_for(config)
    n_sp, n_all = config.n_species, config.n_species + config.n_background
    rows_at = pred_rows(ds, n_all)
    n = staged_rows.shape[0]
    sp = state.species_id[:n]
    num = np.zeros((rows_at[-1], ds.width), dtype=np.float64)
    den = np.zeros(rows_at[-1], dtype=np.float64)
    pp = np.zeros((rows_at[-1], ds.width), dtype=np.float64)
    bkg = np.flatnonzero((sp >= n_sp) & (sp < n_all))
    if len(bkg):
        vals = _values(ds, state, config)[:n]
        _accumulate(ds, rows_at, sp[bkg], ds.stage(sp[bkg], vals[bkg]), state.abundance[bkg],
                    staged_rows[bkg], np.ones(len(bkg), bool), np.ones(len(bkg), bool),
                    num, den, pp)  # fmt: skip
    focal = sp < n_sp
    return sp[focal].copy(), staged_rows[focal].copy(), (num, den, pp)


def _accumulate_numpy(r, abundance, rows, in_diet, in_pp, num, den, pp):
    tot = np.cumsum(rows, axis=1)[:, -1] if rows.shape[1] else np.zeros(len(r))
    ate = in_diet & (tot > 0)
    if ate.any():
        frac = rows[ate] / tot[ate][:, None]
        np.add.at(num, r[ate], abundance[ate][:, None] * frac)
        np.add.at(den, r[ate], abundance[ate])
    if in_pp.any():
        np.add.at(pp, r[in_pp], rows[in_pp])


_NUMBA_KERNEL: list = []  # compiled lazily: the C# bridge path must not import numba


def _numba_kernel():
    """The Numba loop, or None without Numba (compiled on first use)."""
    if not _NUMBA_KERNEL:
        try:
            from numba import njit
        except ImportError:  # pragma: no cover
            _NUMBA_KERNEL.append(None)
            return None

        @njit(cache=True)
        def kernel(r, abundance, rows, in_diet, in_pp, num, den, pp):  # pragma: no cover
            n, w = rows.shape
            for i in range(n):
                tot = 0.0
                for c in range(w):
                    tot += rows[i, c]
                k = r[i]
                if in_diet[i] and tot > 0:
                    a = abundance[i]
                    for c in range(w):
                        num[k, c] += a * (rows[i, c] / tot)
                    den[k] += a
                if in_pp[i]:
                    for c in range(w):
                        pp[k, c] += rows[i, c]

        _NUMBA_KERNEL.append(kernel)
    return _NUMBA_KERNEL[0]


def _accumulate(ds, rows_at, sp, stage, abundance, rows, in_diet, in_pp, num, den, pp):
    """Java's update for these schools, in school order: a school's total is summed left to
    right, its fraction is ``row / total``, weighted by its abundance (C# does the same)."""
    r = (rows_at[sp] + stage).astype(np.int64)
    args = (
        r,
        np.ascontiguousarray(abundance, dtype=np.float64),
        np.ascontiguousarray(rows, dtype=np.float64),
        np.ascontiguousarray(in_diet, dtype=np.bool_),
        np.ascontiguousarray(in_pp, dtype=np.bool_),
        num,
        den,
        pp,
    )
    kernel = _numba_kernel()
    if kernel is not None:
        kernel(*args)
    else:
        _accumulate_numpy(*args)


def collect(state, config, snap) -> dict:
    """The step's staged diet: focal predators staged by their END-of-step values (Java updates
    the outputs after growth and reproduction); diet over present schools that ate, predator
    pressure over alive schools; plus the background contributions of the snapshot."""
    ds = stages_for(config)
    sid, rows, (num, den, pp) = snap
    num, den, pp = num.copy(), den.copy(), pp.copy()
    n = len(sid)
    # the snapshot's focal schools, in order (reproduction only appends; a state that still
    # holds background schools, as a test's may, has them skipped like the C# collector does)
    sub = np.flatnonzero(state.species_id < config.n_species)[:n]
    if len(sub) != n or not np.array_equal(state.species_id[sub], sid):
        raise RuntimeError("staged diet: school order changed since the mortality snapshot")
    if n:
        vals = _values(ds, state, config)[sub]
        alive = _alive(state, config)[sub]
        present = alive & ~state.is_out[sub]
        rows_at = pred_rows(ds, config.n_species + config.n_background)
        _accumulate(ds, rows_at, sid, ds.stage(sid, vals), state.abundance[sub], rows,
                    present, alive, num, den, pp)  # fmt: skip
    return {"diet_stage_num": num, "diet_stage_den": den, "predpressure_stage": pp}
