"""Generate a Chunk-C predation-accessibility matrix: clupeid->cod-egg predation.

Reads the deployed OSMOSE accessibility CSV (prey rows x predator cols) and writes a
variant with cod-as-prey accessible to herring and sprat at `strength`; every other cell
is unchanged. The herring/sprat size-ratio window ([5,500]) restricts this predation to
egg/larval cod automatically, so no explicit prey stage is needed. See
docs/superpowers/specs/2026-07-09-baltic-chunkc-clupeid-cod-egg-predation-design.md.
"""

from __future__ import annotations

import argparse

import pandas as pd

_PREY = "cod"
_PREDATORS = ("herring", "sprat")


def write_chunkc_matrix(deployed_csv: str, strength: float, out_path: str) -> str:
    """cod->herring and cod->sprat set to `strength`; all other cells identical to deployed."""
    df = pd.read_csv(deployed_csv, sep=";", index_col=0)
    if _PREY not in df.index:
        raise KeyError(f"prey row {_PREY!r} not in accessibility matrix {deployed_csv}")
    missing = [p for p in _PREDATORS if p not in df.columns]
    if missing:
        raise KeyError(f"predator column(s) {missing} not in accessibility matrix {deployed_csv}")
    for pred in _PREDATORS:
        df.loc[_PREY, pred] = float(strength)
    df.to_csv(out_path, sep=";")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write a Chunk-C clupeid->cod-egg accessibility matrix"
    )
    ap.add_argument("--deployed", required=True, help="deployed predation-accessibility.csv")
    ap.add_argument(
        "--strength", type=float, required=True, help="cod->herring/sprat accessibility"
    )
    ap.add_argument("--out", required=True, help="output variant CSV path")
    args = ap.parse_args(argv)
    print(write_chunkc_matrix(args.deployed, args.strength, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
