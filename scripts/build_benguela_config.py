"""Synthesize data/benguela/benguela_all-parameters.csv from the external source clone, applying the
Benguela-bundling edit set (seeding, merged forcing, converted maps, fishing stripped). argv[1]=src."""
from __future__ import annotations
import re
import shutil
import sys
from pathlib import Path


def _flatten(master: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    def read(p: Path):
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            parts = re.split(r"\s*[;=]\s*", s, maxsplit=1)
            if len(parts) != 2:
                continue
            k, v = parts[0].strip().lower(), parts[1].strip()
            out[k] = v
            if k.startswith("osmose.configuration.") and (p.parent / v).exists():
                read(p.parent / v)
    read(master)
    return out


def _lines(path: Path) -> dict[str, str]:
    d = {}
    for ln in path.read_text().splitlines():
        if ";" in ln:
            k, v = ln.split(";", 1)
            d[k.strip().lower()] = v.strip()
    return d


def build_config(src_dir: Path, bundle_dir: Path) -> Path:
    raw = _flatten(src_dir / "osmose-ben_seeding.R")
    drop_exact = {"population.initialization.file", "osmose.configuration.initialization",
                  "fisheries.catchability.file", "fisheries.discards.file"}
    for k in list(raw):
        if (k in drop_exact
                or (k.startswith("movement.") and ".map" in k)
                or k.startswith("fisheries.movement.")
                or k.startswith("osmose.configuration.")
                or re.match(r"fisheries\.seasonality\.file\.fsh\d+$", k)):
            del raw[k]
    raw.update({
        "species.file.sp300": "input/roms_climatological_merged.nc",
        "species.file.sp301": "input/roms_climatological_merged.nc",
        "species.file.sp302": "input/roms_climatological_merged.nc",
        "species.file.sp303": "input/roms_climatological_merged.nc",
        "fisheries.enabled": "FALSE",
        "simulation.fishing.mortality.enabled": "FALSE",
        "simulation.nfisheries": "0",
        "output.file.prefix": "benguela",
        "population.seeding.year.max": "30",
        "simulation.time.nyear": "50",
    })
    raw.update(_lines(bundle_dir / "_seeding_keys.txt"))
    raw.update(_lines(bundle_dir / "_movement_keys.txt"))
    idir = bundle_dir / "input"; idir.mkdir(parents=True, exist_ok=True)
    for f in (src_dir / "input").glob("*.csv"):
        shutil.copy(f, idir / f.name)
    shutil.copy(src_dir / "input" / "grid-mask.nc", idir / "grid-mask.nc")
    rep = src_dir / "input" / "reproduction"
    if rep.exists():
        shutil.copytree(rep, idir / "reproduction", dirs_exist_ok=True)
    master = bundle_dir / "benguela_all-parameters.csv"
    master.write_text("\n".join(f"{k} ; {v}" for k, v in sorted(raw.items())) + "\n")
    return master


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1])
    p = build_config(src, root / "data" / "benguela")
    print(f"wrote {p}")
