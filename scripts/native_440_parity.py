"""Round-trip parity gate for the native-4.4.0 cutover (C1).

For each bundled config, compares the Python engine's outputs (biomass/abundance/yield, fixed RNG)
between the OLD 4.3.3 source (captured as a baseline BEFORE conversion) and the NEW native-4.4.0
source (after conversion). The larval-rate x ndt / / ndt round-trip is ~1 ULP, so the gate is a
TIGHT RELATIVE TOLERANCE (default 1e-9), not bit-exact.

  python scripts/native_440_parity.py capture <name>   # run on the CURRENT source, save baseline
  python scripts/native_440_parity.py gate <name>       # run again, assert within tolerance

Also provides `bob-loadpath` (A4.2): a load-path-equivalence gate isolating the Bay-of-Biscay
config KEY conversion (ltl.* -> native species.*) from the forcing resample. It builds a
test-only intermediate (Task 3's original ltl config repointed at the Task 2 24-step file) and
compares it bit-exact against the native config, both reading the identical 24-step forcing.

  python scripts/native_440_parity.py bob-loadpath      # asserts max abs diff == 0.0
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "_parity_baselines"  # gitignored scratch
IN_SCOPE = ("eec_full", "minimal", "baltic", "baltic_ev")
_METRICS = ("biomass", "abundance", "yield")


def _values(df) -> np.ndarray:
    """Stable numeric array from a long-form (time, species, <value>) DataFrame."""
    key_cols = [c for c in ("time", "species") if c in df.columns]
    if key_cols:
        df = df.sort_values(key_cols)
    val_cols = [c for c in df.columns if c not in ("time", "species")]
    return np.asarray(df[val_cols].to_numpy(), dtype=float)


def run_outputs(config_dir, years: int = 3, seed: int = 42) -> dict[str, np.ndarray]:
    """Run the Python engine (fixed seed) and return numeric output arrays."""
    master = next(iter(Path(config_dir).glob("*all-parameters*.csv")))
    raw = dict(OsmoseConfigReader().read(str(master)))
    raw["simulation.time.nyear"] = str(years)
    raw["simulation.rng.fixed"] = "true"
    res = PythonEngine().run_in_memory(raw, seed=seed)  # seed drives determinism
    out: dict[str, np.ndarray] = {}
    accessors = {"biomass": res.biomass, "abundance": res.abundance, "yield": res.yield_biomass}
    for name in _METRICS:
        try:
            df = accessors[name]()
            if df is not None and len(df):
                out[name] = _values(df)
        except Exception as exc:  # metric absent for this config -> skip
            print(f"  ({name} unavailable for {Path(config_dir).name}: {type(exc).__name__})")
    return out


def max_rel_diff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), 1e-30)
    return float(np.nanmax(np.abs(a - b) / denom))


def _build_ltl_24_intermediate(dst: Path) -> Path:
    """Copy the ORIGINAL ltl config (examples_433_orig) and repoint ltl.netcdf.file at the
    24-step file, so the Python _load_config_ltl path reads the same forcing the native config
    does."""
    src = ROOT / "data" / "examples_433_orig"
    shutil.copytree(src, dst)
    ltl = dst / "osm_param-ltl.csv"
    lines = []
    for ln in ltl.read_text().splitlines(keepends=True):
        if ln.strip().lower().startswith("ltl.netcdf.file"):
            lines.append("ltl.netcdf.file ; ltl/roms_n2p2z2d2_biscay_24step.nc\n")
        else:
            lines.append(ln)
    ltl.write_text("".join(lines))
    # bring the 24-step NetCDF into the intermediate's ltl/ dir (it's the Task 5 20x20 regrid,
    # same file the native config points at -- so any diff is a key-conversion defect, not a
    # forcing-resample artifact).
    shutil.copy(
        ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay_24step.nc",
        dst / "ltl" / "roms_n2p2z2d2_biscay_24step.nc",
    )
    return dst


def bob_loadpath_equiv(years: int = 3, seed: int = 42) -> float:
    """Max abs diff (biomass/abundance/yield) between the ltl-24 intermediate and native-24 BoB.

    Both read identical 24-step forcing; a nonzero result is a key-conversion defect.
    """
    native = ROOT / "data" / "examples"  # already native after Task 4
    with tempfile.TemporaryDirectory() as td:
        inter = _build_ltl_24_intermediate(Path(td) / "examples")
        a = run_outputs(inter, years=years, seed=seed)
        b = run_outputs(native, years=years, seed=seed)
    # Guard against a vacuous "0.0 == bit-exact" pass: if run_outputs silently produced no
    # metrics for either config (its per-metric try/except swallows failures), the intersection
    # would be empty and worst would stay 0.0 without ever comparing anything.
    common = set(a) & set(b)
    if not common:
        raise RuntimeError(
            f"no comparable metrics produced (a={sorted(a)}, b={sorted(b)}) — gate cannot validate"
        )
    if set(a) != set(b):
        raise RuntimeError(f"metric set mismatch: a={sorted(a)} vs b={sorted(b)}")
    worst = 0.0
    for k in common:
        if a[k].shape != b[k].shape:
            return float("inf")
        worst = max(worst, float(np.nanmax(np.abs(a[k] - b[k])) if a[k].size else 0.0))
    return worst


def capture_baseline(name: str) -> None:
    BASELINE.mkdir(parents=True, exist_ok=True)
    out = run_outputs(ROOT / "data" / name)
    np.savez(BASELINE / f"{name}.npz", **out)
    print(f"baseline captured: {name} (metrics: {sorted(out)})")


def gate(name: str, tol: float = 1e-9) -> None:
    base = np.load(BASELINE / f"{name}.npz")
    now = run_outputs(ROOT / "data" / name)
    worst = max((max_rel_diff(base[k], now[k]) for k in base.files), default=0.0)
    verdict = "PASS" if worst < tol else "FAIL"
    print(f"{name}: max_rel_diff={worst:.2e} {verdict} (tol={tol:.0e})")
    assert worst < tol, f"{name} parity FAILED: {worst:.2e} >= {tol:.0e}"


if __name__ == "__main__":
    cmd = sys.argv[1]  # read cmd FIRST -- bob-loadpath takes no target (avoid IndexError)
    if cmd == "bob-loadpath":
        w = bob_loadpath_equiv()
        print(f"bob load-path equiv: max abs diff = {w:.2e} {'PASS' if w == 0.0 else 'FAIL'}")
        assert w == 0.0, f"BoB key conversion NOT lossless: {w:.2e}"
    else:
        target = sys.argv[2]
        if target not in IN_SCOPE:
            raise SystemExit(f"{target} not in scope {IN_SCOPE}")
        capture_baseline(target) if cmd == "capture" else gate(target)
