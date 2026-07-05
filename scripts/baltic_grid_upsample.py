from pathlib import Path
import numpy as np
import pandas as pd
from osmose.forcing.grid_upsample import block_replicate

PERCID = {"perch", "pikeperch"}
JOBS = [
    (Path("data/baltic/maps"), Path("data/baltic-fine/maps"), True),
    (Path("data/baltic/fishing"), Path("data/baltic-fine/fishing"), False),
]


def main() -> int:
    for src, dst, percid_control in JOBS:
        dst.mkdir(parents=True, exist_ok=True)
        for csv in sorted(src.glob("*.csv")):  # *.csv skips *.pre-mask-rebuild.bak siblings
            arr = pd.read_csv(csv, sep=";", header=None).values.astype(
                float
            )  # south->north on disk
            up = block_replicate(arr, 4)
            sp = csv.stem.split("_")[0]
            name = csv.stem + "_upsampled.csv" if (percid_control and sp in PERCID) else csv.name
            np.savetxt(dst / name, up, fmt="%.4f", delimiter=";")
            print(f"upsampled {csv.name} -> {dst / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
