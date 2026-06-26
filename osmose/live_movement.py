"""Live-during-run movement snapshots + queue transport (Python engine only).

Pure core module — no UI imports. Produces a per-step ``MovementSnapshot`` of living
focal schools' positions for the Run-page live map, and a throttling queue observer the
engine step hook calls. ``queue.Queue`` is stdlib, so the transport helper stays here and
is unit-testable.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.live_movement")

# Ontogenetic life stage per school for the live-movement filter.
STAGE_LABELS = {0: "Egg/larva", 1: "Juvenile", 2: "Adult"}


@dataclass
class MovementSnapshot:
    """One frame of living focal schools' positions for the live map."""

    step: int
    n_steps: int
    # "running" | "done" | "cancelled". Frames from the live observer are always
    # "running"; the Run page shows the terminal status from its own _live_status_val
    # (set on completion/cancel), so a retained final frame stays "running" by design.
    status: str
    species: list[str]
    sp_id: NDArray[np.int32]
    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    biomass: NDArray[np.float64]
    stage: NDArray[np.int8]  # per-school life stage: 0=egg/larva, 1=juvenile, 2=adult
    truncated: bool
    n_total: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_step: float  # grid cell spacing (median diff of full lon array); 0 if < 2 cells
    lat_step: float


def resolve_grid_latlon(grid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(lat, lon)`` cell-coordinate arrays.

    Uses ``grid.lat``/``grid.lon`` when present (NetCDF-backed grids), else
    ``np.arange(ny|nx, dtype=float64)`` — byte-identical to the spatial-NetCDF writer's
    fallback (``osmose/engine/output.py:726,731``), so live and post-run coords agree.
    """
    lat = grid.lat if grid.lat is not None else np.arange(grid.ny, dtype=np.float64)
    lon = grid.lon if grid.lon is not None else np.arange(grid.nx, dtype=np.float64)
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def build_snapshot(
    step: int, state, grid, config, *, status: str = "running", dot_cap: int = 2000
) -> MovementSnapshot:
    """Build a snapshot of focal + located + living schools at ``step`` (pure).

    Selection mask = focal (``species_id < n_species``) & in-domain (``~is_out``) &
    located (``cell_x/cell_y >= 0``, drops freshly-spawned eggs at ``-1``) & living
    (``biomass > 0``). Samples to ``dot_cap`` (default 2000) deterministically when
    exceeded.
    """
    lat_arr, lon_arr = resolve_grid_latlon(grid)
    mask = (
        (state.species_id < config.n_species)
        & ~state.is_out
        & (state.cell_x >= 0)
        & (state.cell_y >= 0)
        & (state.biomass > 0.0)
    )
    sp_id = state.species_id[mask]
    cx = state.cell_x[mask]
    cy = state.cell_y[mask]
    bm = state.biomass[mask]
    # Life stage: egg/larva if is_egg; adult if mature (length>=maturity_size AND
    # age>=maturity_age, per reproduction.py); else juvenile. Computed on the mask.
    is_egg_m = state.is_egg[mask]
    mature_m = (state.length[mask] >= config.maturity_size[sp_id]) & (
        state.age_dt[mask] >= config.maturity_age_dt[sp_id]
    )
    stage = np.where(is_egg_m, 0, np.where(mature_m, 2, 1)).astype(np.int8)
    n_total = int(sp_id.size)
    truncated = n_total > dot_cap
    if truncated:
        idx = np.linspace(0, n_total - 1, dot_cap).astype(np.intp)
        sp_id, cx, cy, bm, stage = sp_id[idx], cx[idx], cy[idx], bm[idx], stage[idx]
    lon_step = float(np.median(np.diff(lon_arr))) if lon_arr.size > 1 else 0.0
    lat_step = float(np.median(np.diff(lat_arr))) if lat_arr.size > 1 else 0.0
    return MovementSnapshot(
        step=int(step),
        n_steps=int(config.n_steps),
        status=status,
        species=list(config.species_names[: config.n_species]),
        sp_id=np.asarray(sp_id, dtype=np.int32),
        lon=lon_arr[cx].astype(np.float64),
        lat=lat_arr[cy].astype(np.float64),
        biomass=np.asarray(bm, dtype=np.float64),
        stage=np.asarray(stage, dtype=np.int8),
        truncated=truncated,
        n_total=n_total,
        lon_min=float(lon_arr.min()),
        lon_max=float(lon_arr.max()),
        lat_min=float(lat_arr.min()),
        lat_max=float(lat_arr.max()),
        lon_step=lon_step,
        lat_step=lat_step,
    )


def make_step_observer(
    q: "queue.Queue[MovementSnapshot]",
    *,
    dot_cap: int = 2000,
    throttle_s: float = 0.2,
    now: Callable[[], float] = time.monotonic,
) -> Callable[[int, object, object, object], None]:
    """Return a step-observer that builds a snapshot and enqueues it (drop-oldest).

    Always emits step 0 and the final step (``config.n_steps - 1``); throttles the
    rest by wall-clock. ``dot_cap`` (default 2000) is passed through to
    ``build_snapshot`` and bounds the points streamed to the client per frame. Never
    blocks the engine thread and never raises into it.
    """
    last_emit: list[float | None] = [None]

    def observer(step: int, state, grid, config) -> None:
        n_steps = int(config.n_steps)
        is_edge = step == 0 or step == n_steps - 1
        t = now()
        if not is_edge and last_emit[0] is not None and (t - last_emit[0]) < throttle_s:
            return
        last_emit[0] = t
        try:
            snap = build_snapshot(step, state, grid, config, dot_cap=dot_cap)
        except Exception:  # noqa: BLE001 — never crash the running simulation
            _log.warning("live snapshot build failed at step %s", step, exc_info=True)
            return
        try:
            q.put_nowait(snap)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(snap)
            except queue.Full:
                pass

    return observer


def make_run_observer(
    progress_q: "queue.Queue[tuple[int, int, float]]",
    live_observer: "Callable[[int, object, object, object], None] | None" = None,
    *,
    now: Callable[[], float] = time.monotonic,
) -> Callable[[int, object, object, object], None]:
    """Step-observer that pushes (done, n_steps, elapsed_s) to progress_q every step
    (done = step + 1, 1-based) and delegates to live_observer when given.

    Never raises into the engine. Progress is pushed BEFORE delegating, so a failing
    live_observer cannot suppress progress. Drop-oldest on a maxsize-1 queue.
    """
    start: list[float | None] = [None]

    def observer(step: int, state, grid, config) -> None:
        try:
            if start[0] is None:
                start[0] = now()
            done = step + 1
            n = int(config.n_steps)
            elapsed = now() - start[0]
            try:
                progress_q.put_nowait((done, n, elapsed))
            except queue.Full:
                try:
                    progress_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    progress_q.put_nowait((done, n, elapsed))
                except queue.Full:
                    pass
            if live_observer is not None:
                live_observer(step, state, grid, config)
        except Exception:  # noqa: BLE001 — never crash the running simulation
            _log.warning("run observer failed at step %s", step, exc_info=True)

    return observer


def config_is_spatial(config: dict[str, str]) -> bool:
    """True when the config has a regular grid that yields live-movement frames
    (GridSpec.from_config succeeds — needs grid.nlon/nlat/upleft.*/lowright.*).
    Configs lacking those regular-grid keys (e.g. NcGrid configs that specify only
    grid.netcdf.file) -> False.
    """
    try:
        GridSpec.from_config(config)
        return True
    except (KeyError, ValueError, TypeError):
        return False


def format_progress_label(done: int, n_steps: int, ndt: int) -> str:
    """Human progress label from 1-based completed-step count `done`.

    ndt > 0 -> 'Year y/ny · step done/n · pct%' (year bucket uses (done-1)//ndt to
    convert the 1-based done back to a 0-based index, so a 1-year run's final tick
    done==ndt gives Year 1/1, not Year 2/1). ndt <= 0 -> step-only label (no division).
    """
    pct = round(done / n_steps * 100) if n_steps else 0
    if ndt and ndt > 0:
        year = (done - 1) // ndt + 1
        n_years = -(-n_steps // ndt)  # ceil division
        return f"Year {year}/{n_years} · step {done}/{n_steps} · {pct}%"
    return f"Step {done}/{n_steps} · {pct}%"
