#!/usr/bin/env python3
"""Derive ICES catch-based CATCH calibration targets from the in-repo snapshot.

One-shot: reads data/baltic/reference/ices_snapshots/, sums each model species' stock
catches (falling back to landings where catches are unreported) over 2018-2022, and emits
`catch` target rows (band = mean +/- 1.5*std, floored at the window min). Writes
biomass_targets.csv in place, preserving comment/provenance lines.

Run: .venv/bin/python scripts/derive_ices_targets.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO / "data" / "baltic" / "reference" / "ices_snapshots"
TARGETS_CSV = REPO / "data" / "baltic" / "reference" / "biomass_targets.csv"
WINDOW = (2018, 2022)
K_STD = 1.5
CATCH_WEIGHT = 0.5
ASSESSED = ("cod", "herring", "sprat", "flounder")


def _stock_catch(snapshot_dir: Path, stock: str) -> dict[int, float]:
    """year -> catch (tonnes) for a stock, complete years only.

    Prefers ICES `catches` (landings + discards, the correct analog to the model's
    total-fished-biomass `yield`); falls back to `landings` for years/stocks where
    `catches` is unreported (empty string or missing).
    """
    recs = json.load(open(snapshot_dir / f"{stock}.assessment.json"))
    out: dict[int, float] = {}
    for r in recs:
        y = r.get("year")
        catch = r.get("catches") or r.get("landings")
        if y and catch not in (None, ""):
            out[int(y)] = float(catch)
    return out


def derive_catch_targets(snapshot_dir: Path) -> list[dict]:
    """One catch-target row dict per assessed species (catches summed across its stocks)."""
    index = json.load(open(snapshot_dir / "index.json"))
    mapping = index["model_species_to_ices_stocks"]
    lo_y, hi_y = WINDOW
    rows: list[dict] = []
    for sp in ASSESSED:
        # Sum catches across the species' stocks per year (only years present contribute).
        per_year: dict[int, float] = {}
        for stock in mapping[sp]:
            for y, catch in _stock_catch(snapshot_dir, stock).items():
                if lo_y <= y <= hi_y:
                    per_year[y] = per_year.get(y, 0.0) + catch
        vals = np.array([per_year[y] for y in sorted(per_year)], dtype=float)
        if vals.size == 0:
            raise ValueError(f"no catches in window {WINDOW} for {sp}")
        mean, std, vmin = float(vals.mean()), float(vals.std()), float(vals.min())
        lower = max(mean - K_STD * std, vmin)
        upper = mean + K_STD * std
        rows.append(
            {
                "species": sp,
                # Full precision (not rounded to whole tonnes): rounding here would lose the
                # sub-tonne precision that test_sprat_catch_matches_snapshot_mean checks to
                # rel=1e-9 against the raw ICES catches (fallback landings) mean.
                "target_tonnes": str(mean),
                "lower_tonnes": str(lower),
                "upper_tonnes": str(upper),
                "weight": f"{CATCH_WEIGHT}",
                "reference_point_type": "catch",
                # NOTE: these are written into an unquoted CSV (see _rewrite_csv) parsed by
                # csv.DictReader downstream — any literal "," here would silently shift
                # columns. Use "; " (matching the file's existing convention) instead.
                "source": (
                    f"ICES catches (fallback landings) {WINDOW[0]}-{WINDOW[1]} summed over "
                    f"{'; '.join(mapping[sp])}"
                ),
                "notes": f"mean+/-{K_STD}sigma; floored at window min ({vals.size} yr)",
            }
        )
    return rows


def _rewrite_csv(catch_rows: list[dict]) -> None:
    """Preserve comment/header lines + existing biomass data rows; append catch rows; bump version."""
    text = TARGETS_CSV.read_text().splitlines()
    header_idx = next(i for i, ln in enumerate(text) if ln.startswith("species,"))
    comments = text[:header_idx]
    header = text[header_idx]
    cols = header.split(",")
    data_rows = [ln for ln in text[header_idx + 1 :] if ln.strip()]
    # Drop any pre-existing catch rows (idempotent re-run).
    data_rows = [ln for ln in data_rows if ",catch," not in f",{ln},"]
    today = date.today().isoformat()
    comments = [
        (f"#! last_updated: {today}" if ln.startswith("#! last_updated:") else ln)
        for ln in comments
    ]
    for r in catch_rows:
        for c in cols:
            v = r.get(c, "")
            if "," in v:
                raise ValueError(
                    f"catch row field {c!r}={v!r} contains a literal comma; this file is "
                    f"written/read as unquoted CSV (csv.DictReader) — use ';' instead"
                )
    new_rows = [",".join(r.get(c, "") for c in cols) for r in catch_rows]
    TARGETS_CSV.write_text("\n".join([*comments, header, *data_rows, *new_rows]) + "\n")


def main() -> None:
    rows = derive_catch_targets(SNAPSHOT_DIR)
    _rewrite_csv(rows)
    print(f"Wrote {len(rows)} catch rows to {TARGETS_CSV}")
    for r in rows:
        print(
            f"  {r['species']:9} catch target={r['target_tonnes']} "
            f"[{r['lower_tonnes']}, {r['upper_tonnes']}] t"
        )


if __name__ == "__main__":
    main()
