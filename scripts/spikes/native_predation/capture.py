"""Capture the production cell-loop kernel's pre-state from a live eec_full run.

The leaf is called from inside @njit and cannot be intercepted directly, but
its args are (almost all) the arrays the cell-loop kernel receives — which IS
patchable because mortality() dispatches to it by module-global name.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid as G
from osmose.engine.processes import mortality as M
from osmose.engine.simulate import simulate

from .provenance import assert_provenance


def capture_cellloop(config_path: Path, capture_call_index: int, out_dir: Path,
                     worktree_root: Path) -> Path:
    info = assert_provenance(worktree_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_fn = M._mortality_all_cells_parallel
    params = list(inspect.signature(real_fn.py_func).parameters)  # njit -> .py_func
    state: dict = {"n": 0, "captured": None}

    def wrapper(*args):
        if state["n"] == capture_call_index and state["captured"] is None:
            state["captured"] = {
                name: (np.copy(a) if isinstance(a, np.ndarray) else a)
                for name, a in zip(params, args)
            }
        state["n"] += 1
        return real_fn(*args)

    M._mortality_all_cells_parallel = wrapper
    try:
        reader = OsmoseConfigReader()
        raw = reader.read(config_path)
        raw["simulation.time.nyear"] = "2"  # 24 dt/yr x 2 x subdt(10) = 480 kernel calls > 200
        # Calibration workload: diet output OFF (spec §4.0). Calibration skips diet
        # aggregation (output-gated), so disabling it makes the captured `diet_enabled`
        # scalar False and measures the leaf as calibration exercises it.
        # NOTE: `tl_tracking` CANNOT be forced off from config — the engine sets
        # ctx.tl_weighted_sum unconditionally whenever a ctx exists (mortality.py:1824),
        # so the captured `tl_tracking` scalar is always True. We accept TL-on; the C port
        # handles both branches and the real flag value is recorded in meta.json.
        raw["output.diet.composition.enabled"] = "false"
        cfg = EngineConfig.from_dict(raw)
        grid = G.from_netcdf(config_path.parent / raw["grid.netcdf.file"],
                             mask_var=raw.get("grid.var.mask", "mask"))
        simulate(cfg, grid, np.random.default_rng(42))
    finally:
        M._mortality_all_cells_parallel = real_fn

    cap = state["captured"]
    if cap is None:
        raise RuntimeError(f"capture_call_index={capture_call_index} never reached "
                           f"(only {state['n']} cell-loop calls)")

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {}

    for k, v in cap.items():
        if isinstance(v, np.ndarray) and v.ndim > 0:
            arrays[k] = v
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            # 0-d numpy array: route to scalars (deviation from brief — cast via .item())
            scalars[k] = v.item()
        else:
            scalars[k] = v

    npz_path = out_dir / "cellloop.npz"
    np.savez(npz_path, **arrays)

    def _json_scalar(v: object) -> int | float | bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return float(v)  # type: ignore[arg-type]

    meta = {
        "provenance": info,
        "arg_order": params,
        "scalars": {k: _json_scalar(v) for k, v in scalars.items()},
        "n_resources": int(scalars.get("n_resources",
                                       arrays["rsc_biomass"].shape[0])),
        "n_cells": int(len(arrays["boundaries"]) - 1),
        "flags": {
            "diet_enabled": bool(scalars.get("diet_enabled", False)),
            "tl_tracking": bool(scalars.get("tl_tracking", False)),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return npz_path


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[3]
    cfg = root / "data" / "eec_full" / "eec_all-parameters.csv"
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    p = capture_cellloop(cfg, idx, root / "scripts/spikes/native_predation/_fixtures", root)
    print("captured ->", p)
