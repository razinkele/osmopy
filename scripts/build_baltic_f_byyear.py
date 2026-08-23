#!/usr/bin/env python
"""Derive by-year fishing-mortality CSVs for the F1 hindcast (spec 2026-08-23).

Offline: reads data/baltic/reference/ices_snapshots/*.assessment.json (cached; no
network) and data/baltic/baltic_param-fishing.csv (base F, verbatim strings).
Writes data/baltic/reference/f_byyear_sp{0,1,2,8}.csv — 50 rows each: 19 spin-up
rows carrying the base-F string verbatim (arms must share the pre-period
bit-exactly), then 31 rows base_F * factor(1993..2023).

Scaling (spec decisions 2-3): factor_s(y) = F_s(y) / mean(F_s over available
years in 2018-2022); herring aggregates the four stocks' FACTORS (scale-free)
with per-year catch weights in tonnes. Flounder (sp3) gets NO file (decision 5:
its calibrated base F is 6.4x its ICES anchor — incommensurable). cod_west F
ends 2021 -> hold-last.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data/baltic/reference/ices_snapshots"
FISHING_CSV = ROOT / "data/baltic/baltic_param-fishing.csv"
OUT_DIR = ROOT / "data/baltic/reference"

YEARS = list(range(1993, 2024))  # 31 hindcast years
SPINUP = 19                      # sim-years 0-18; sim-year 19 = 1993
ANCHOR = (2018, 2022)

STOCKS: dict[int, tuple[str, list[str]]] = {
    0: ("cod_west", ["cod.27.22-24"]),
    1: ("herring", ["her.27.25-2932", "her.27.28", "her.27.3031", "her.27.20-24"]),
    2: ("sprat", ["spr.27.22-32"]),
    8: ("cod_east", ["cod.27.24-32"]),
}


def load_stock(snap_dir: Path, stock_key: str) -> tuple[dict[int, float], dict[int, float]]:
    """(F-by-year, catches-by-year); snapshot values are strings, '' = missing."""
    recs = json.loads((snap_dir / f"{stock_key}.assessment.json").read_text())
    f = {int(r["year"]): float(r["f"]) for r in recs if r.get("f") not in ("", None)}
    c = {int(r["year"]): float(r["catches"]) for r in recs if r.get("catches") not in ("", None)}
    return f, c


def hold_last(series: dict[int, float], years: list[int]) -> list[float]:
    out: list[float] = []
    last: float | None = None
    for y in years:
        if y in series:
            last = series[y]
        if last is None:
            raise ValueError(f"no value at or before {y}")
        out.append(last)
    return out


def anchor_mean(f: dict[int, float]) -> float:
    vals = [f[y] for y in range(ANCHOR[0], ANCHOR[1] + 1) if y in f]
    if not vals:
        raise ValueError(f"no F values in anchor window {ANCHOR}")
    return sum(vals) / len(vals)


def factor_series(f: dict[int, float]) -> list[float]:
    a = anchor_mean(f)
    return [v / a for v in hold_last(f, YEARS)]


def herring_factor_series(stocks: list[tuple[dict[int, float], dict[int, float]]]) -> list[float]:
    per_stock = [factor_series(f) for f, _ in stocks]
    weights = [hold_last(c, YEARS) for _, c in stocks]
    out: list[float] = []
    for i in range(len(YEARS)):
        w = [wt[i] for wt in weights]
        out.append(sum(wi * fs[i] for wi, fs in zip(w, per_stock)) / sum(w))
    return out


def read_base_f_strings(fishing_csv: Path) -> dict[int, str]:
    """Raw base-F strings by species index. Relies on the identity sp<->fsh mapping
    (data/baltic/fishery-catchability.csv)."""
    out: dict[int, str] = {}
    for line in fishing_csv.read_text().splitlines():
        if line.startswith("fisheries.rate.base.fsh"):
            key, val = line.split(";", 1)
            out[int(key.rsplit("fsh", 1)[1])] = val.strip()
    return out


def build_rows(base_str: str, factors: list[float]) -> list[str]:
    base = float(base_str)
    # repr() is the shortest round-trip representation: float(repr(x)) == x exactly,
    # so the scaled rows lose no precision through np.loadtxt.
    return [base_str] * SPINUP + [repr(base * f) for f in factors]


def write_csv(path: Path, rows: list[str], header_lines: list[str]) -> None:
    text = "".join(f"# {h}\n" for h in header_lines) + "\n".join(rows) + "\n"
    path.write_text(text)


def main() -> None:
    base_strings = read_base_f_strings(FISHING_CSV)
    for sp_idx, (name, stock_keys) in STOCKS.items():
        loaded = [load_stock(SNAP, k) for k in stock_keys]
        factors = (
            herring_factor_series(loaded) if len(loaded) > 1 else factor_series(loaded[0][0])
        )
        rows = build_rows(base_strings[sp_idx], factors)
        header = [
            f"F1 hindcast by-year F for {name} (sp{sp_idx}) — generated {date.today()}",
            f"stocks: {', '.join(stock_keys)}; anchor: mean F over available years "
            f"{ANCHOR[0]}-{ANCHOR[1]}; base F (verbatim): {base_strings[sp_idx]}",
            f"layout: {SPINUP} spin-up rows at base F, then {len(YEARS)} rows "
            f"base*factor for {YEARS[0]}-{YEARS[-1]} (sim-year {SPINUP} = {YEARS[0]})",
            f"factor range: {min(factors):.3g}-{max(factors):.3g}",
            "spec: docs/superpowers/specs/2026-08-23-baltic-f1-historical-fishing-hindcast-design.md",
        ]
        out = OUT_DIR / f"f_byyear_sp{sp_idx}.csv"
        write_csv(out, rows, header)
        print(f"wrote {out} (factor range {min(factors):.3g}-{max(factors):.3g})")


if __name__ == "__main__":
    main()
