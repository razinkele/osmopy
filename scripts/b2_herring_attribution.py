#!/usr/bin/env python
"""Which v1.9.0 change halved herring's B2 warming response?

B2's ``rcp85_ref`` arm (+2.9 °C through the C1 reproduction thermal gate) cut herring's
final-decade biomass by 62.27% in August 2026 and by 26.47% on v1.9.0. Between v1.8.8 and v1.9.0
only data changed (#236): five files, of which the movement file changed in comments alone. Each
arm here reverts some of those files to v1.8.8 (``1d8bd214``) at FILE level in the ``osmose_demo``
copy, before the config is read. A dict override of a larval rate would skip the reader's ÷24
per-step migration. Each arm then runs B2's baseline and ``rcp85_ref`` pair over B2's five house
seeds, and herring's decline is computed exactly as ``run_b2`` computes it.

Arms:
  A0  v1.9.0 as shipped (instrument gate: must reproduce B2's -26.47%)
  A1  herring larval rate sp1 -> v1.8.8 (98.56 /yr)
  A2  herring reproduction seasonality -> v1.8.8 (25% autumn share)
  A3  LTL forcing -> v1.8.8 (pre-#234 column integral)
  A4  every #236 data file -> v1.8.8 (A1 + A2 + A3 + A5)
  A5  the cod-only pair -> v1.8.8 (cod_east SD 25 spawning map, cod_west larval sp0)

Pre-stated criterion: ``docs/diagnostics/b2_herring_attribution_prestated_2026-11-11.md``.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/b2_herring_attribution.py \
        --out docs/diagnostics/b2_herring_attribution_2026-11-11.json
"""

from __future__ import annotations

import os

# Single-threaded Numba per worker; must be set before numba is imported anywhere.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V188 = "1d8bd214"
SEEDS = (42, 123, 7, 999, 2024)
D_AUGUST = -0.6227418530925009  # rcp85_ref, docs/diagnostics/baltic_b2_scenario_report.json
D_B2_V190 = (
    -0.26469846771759054
)  # rcp85_ref, docs/diagnostics/baltic_b2_scenario_report_2026-11-11.json

F_LTL = "baltic_ltl_biomass.nc"
F_MORT = "baltic_param-additional-mortality.csv"
F_SEASON = "reproduction/reproduction-seasonality-sp1.csv"
F_CODMAP = "maps/cod_east_spawning.csv"
LARVA_KEY = "mortality.additional.larva.rate.sp{i}"

# arm -> (whole files to restore from v1.8.8, larval-rate species whose CSV line is restored)
ARMS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "A0": ((), ()),
    "A1": ((), (1,)),
    "A2": ((F_SEASON,), ()),
    "A3": ((F_LTL,), ()),
    "A4": ((F_LTL, F_SEASON, F_CODMAP), (0, 1)),
    "A5": ((F_CODMAP,), (0,)),
}


def _git_show(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{V188}:data/baltic/{path}"])


def _larva_line(text: str, sp: int) -> str:
    key = LARVA_KEY.format(i=sp)
    lines = [ln for ln in text.splitlines() if ln.split(";", 1)[0].strip() == key]
    assert len(lines) == 1, (key, lines)
    return lines[0]


def prepare_arm_dir(arm: str, v188: dict[str, bytes], dest: Path) -> Path:
    """osmose_demo copy of the current data, with this arm's files reverted to v1.8.8."""
    from osmose.demo import osmose_demo

    config_file = Path(osmose_demo("baltic", dest)["config_file"])
    data_dir = config_file.parent
    files, larva_sps = ARMS[arm]
    for f in files:
        (data_dir / f).write_bytes(v188[f])
    if larva_sps:
        mort = data_dir / F_MORT
        text = mort.read_text()
        old_text = v188[F_MORT].decode()
        for sp in larva_sps:
            text = text.replace(_larva_line(text, sp), _larva_line(old_text, sp))
        mort.write_text(text)
    return config_file


def expected_resolved(arm: str, v188: dict[str, bytes], current_mort: str) -> dict[int, float]:
    """Resolved (per-step) larval rate each arm must reach the engine with, for sp0 and sp1."""
    _files, larva_sps = ARMS[arm]
    out = {}
    for sp in (0, 1):
        src = v188[F_MORT].decode() if sp in larva_sps else current_mort
        out[sp] = float(_larva_line(src, sp).split(";")[1]) / 24.0
    return out


def run_one(job: tuple) -> dict:
    arm, variant, seed, v188, overlays, current_mort = job
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix=f"attr_{arm}_{variant}_{seed}_"))
    try:
        cfg = dict(OsmoseConfigReader().read(str(prepare_arm_dir(arm, v188, tmp))))
        # The revert must reach the engine in resolved (per-step) units: guard against the
        # ÷24 reader migration being skipped or doubled.
        want = expected_resolved(arm, v188, current_mort)
        for sp, v in want.items():
            got = float(cfg[LARVA_KEY.format(i=sp)])
            # The reader writes the migrated rate to 10 significant digits ('10.15686139'),
            # so compare at a relative 1e-8. A skipped or doubled ÷24 is off by 24x.
            assert abs(got - v) <= 1e-8 * max(1.0, abs(v)), (arm, sp, got, v)
        cfg["simulation.time.nyear"] = "50"
        if variant == "rcp85_ref":
            cfg.update(overlays)
        bio = PythonEngine().run_in_memory(cfg, seed=seed).biomass()
        names = [c for c in bio.columns if c not in ("Time", "species")]
        series = {n: bio[n].to_numpy(dtype=float).tolist() for n in names}
        return {
            "arm": arm,
            "variant": variant,
            "seed": seed,
            "resolved_larva": want,
            "series": series,
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def decade_mean(series: list[float], n_year: int = 50) -> float:
    x = np.asarray(series, dtype=float)
    x = x if len(x) == n_year else x.reshape(n_year, -1).mean(axis=1)
    return float(x[-10:].mean())


def analyse(runs: list[dict], herring: str = "herring") -> dict:
    out = {}
    by = {(r["arm"], r["variant"], r["seed"]): r for r in runs}
    a0 = None
    for arm in ARMS:
        base = np.array([decade_mean(by[(arm, "baseline", s)]["series"][herring]) for s in SEEDS])
        warm = np.array([decade_mean(by[(arm, "rcp85_ref", s)]["series"][herring]) for s in SEEDS])
        decline = float(warm.mean() / base.mean() - 1.0)  # exactly run_b2's formula
        paired = warm / base - 1.0
        se = float(paired.std(ddof=1) / np.sqrt(len(SEEDS)))
        out[arm] = {
            "decline": decline,
            "paired_declines": paired.tolist(),
            "se": se,
            "baseline_herring_mean": float(base.mean()),
            "baseline_herring_per_seed": base.tolist(),
        }
        if arm == "A0":
            a0 = out[arm]
    assert a0 is not None
    gap = D_AUGUST - a0["decline"]
    for arm, v in out.items():
        v["fraction_restored"] = (v["decline"] - a0["decline"]) / gap
        comb = float(np.hypot(v["se"], a0["se"]))
        v["z_vs_A0"] = (v["decline"] - a0["decline"]) / comb if comb > 0 else None
    out["_gate"] = {
        "A0_decline": a0["decline"],
        "B2_v190_decline": D_B2_V190,
        "abs_diff": abs(a0["decline"] - D_B2_V190),
        "pass": abs(a0["decline"] - D_B2_V190) <= 0.005,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import baltic_b2_scenario_ab as b2

    spec = json.loads(b2.DEFAULT_SPEC_PATH.read_text())
    arm_def = {a["name"]: a for a in spec["arms"]}["rcp85_ref"]
    art_root = Path(tempfile.mkdtemp(prefix="attr_artifacts_"))
    artifacts = b2._b2.write_arm_dir(
        arm_def,
        art_root / "rcp85_ref",
        b2.DEFAULT_O2_PATH,
        b2.DEFAULT_GRID_PATH,
        b2.TREFS,
        b2.BETAS,
    )
    overlays = b2.arm_overlays("rcp85_ref", artifacts, b2.TREFS, b2.BETAS)
    v188 = {f: _git_show(f) for f in (F_LTL, F_MORT, F_SEASON, F_CODMAP)}
    current_mort = (ROOT / "data/baltic" / F_MORT).read_text()

    jobs = [
        (arm, variant, seed, v188, overlays, current_mort)
        for arm in ARMS
        for variant in ("baseline", "rcp85_ref")
        for seed in SEEDS
    ]
    print(
        f"{len(jobs)} runs x 50 yr, {args.workers} workers; overlays: {sorted(overlays)}",
        flush=True,
    )
    t0 = time.time()
    runs = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(run_one, jobs):
            runs.append(r)
            h = decade_mean(r["series"]["herring"])
            print(
                f"[{time.time() - t0:6.0f}s] {r['arm']} {r['variant']:9s} seed={r['seed']:<5}"
                f" herring final-decade {h:12.0f}  ({r['seconds']}s)",
                flush=True,
            )
    result = {
        "seeds": SEEDS,
        "arms": {k: list(v[0]) + [f"larva sp{i}" for i in v[1]] for k, v in ARMS.items()},
    }
    result["analysis"] = analyse(runs)
    result["runs"] = runs
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    g = result["analysis"]["_gate"]
    print(
        f"GATE A0 {g['A0_decline']:+.4%} vs B2 {g['B2_v190_decline']:+.4%}: {'PASS' if g['pass'] else 'FAIL'}"
    )
    for arm in ARMS:
        v = result["analysis"][arm]
        print(
            f"{arm}: decline {v['decline']:+.2%} (SE {v['se']:.2%})  f={v['fraction_restored']:+.2f}"
            f"  z={v['z_vs_A0']}  baseline herring {v['baseline_herring_mean']:.3g}"
        )
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
