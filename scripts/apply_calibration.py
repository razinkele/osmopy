"""Apply a calibration results JSON to the tracked Baltic config CSVs, in place.

Switches all 8 focal species to Shepherd stock-recruitment and writes every
calibrated mortality / fishing / recruitment parameter into its owning CSV,
editing only the affected key lines so comments and structure are preserved.
Run: .venv/bin/python scripts/apply_calibration.py <results.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_CONFIG_DIR = Path("data/baltic")
_FILE_FOR = {
    "stock.recruitment.": "baltic_param-reproduction.csv",
    "mortality.additional.": "baltic_param-additional-mortality.csv",
    "fisheries.rate.base.": "baltic_param-fishing.csv",
    # Background-predator (cormorant sp15) calibration levers. NOTE: the bare
    # `predation.ingestion.rate.max.` prefix also matches focal sp0-7 (which live
    # in baltic_param-predation.csv) — safe here because only sp15 is a free param;
    # guard this if a focal ingestion is ever added to the DE free set.
    "species.biomass.multiplier.": "baltic_param-background.csv",
    "predation.ingestion.rate.max.": "baltic_param-background.csv",
}
# The reader divides the larval additional-mortality scalar by ndtperyear on read
# (osmose/config/aliases.py); every OTHER key is identity. So the file must store
# authored_value * ndt for larval rate keys and the raw value for all others.
_LARVA_RATE_RE = re.compile(r"^mortality\.additional\.larva\.rate\.sp\d+$")


def _ndtperyear(config_dir: Path) -> int:
    sim = config_dir / "baltic_param-simulation.csv"
    if sim.exists():
        for line in sim.read_text().splitlines():
            s = line.strip()
            if s.lower().startswith("simulation.time.ndtperyear") and ";" in s:
                return int(float(s.split(";", 1)[1].strip()))
    return 24


def _file_for(key: str, config_dir: Path) -> Path:
    for prefix, fname in _FILE_FOR.items():
        if key.startswith(prefix):
            return config_dir / fname
    raise KeyError(f"no tracked CSV owns key {key!r}")


def set_key(path: Path, key: str, value) -> None:
    """Set ``key;value`` in a ``;``-separated OSMOSE CSV, in place; append if absent."""
    lines = path.read_text().splitlines() if path.exists() else []
    out, found = [], False
    for line in lines:
        s = line.strip()
        if (
            s
            and not s.startswith("#")
            and ";" in s
            and s.split(";", 1)[0].strip().lower() == key.lower()
        ):
            out.append(f"{key};{value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key};{value}")
    path.write_text("\n".join(out) + "\n")


def apply_calibration(results_path: Path, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
    params = json.loads(Path(results_path).read_text())["parameters"]
    ndt = _ndtperyear(config_dir)
    repro = config_dir / _FILE_FOR["stock.recruitment."]
    for i in range(8):
        set_key(repro, f"stock.recruitment.type.sp{i}", "shepherd")
    for key, val in params.items():
        write_val = float(val) * ndt if _LARVA_RATE_RE.match(key) else val
        set_key(_file_for(key, config_dir), key, write_val)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a calibration JSON to tracked Baltic CSVs")
    ap.add_argument("results_json")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    args = ap.parse_args()
    cfg_dir = Path(args.config_dir)
    apply_calibration(Path(args.results_json), cfg_dir)

    from osmose.config import OsmoseConfigReader  # roundtrip check

    cfg = OsmoseConfigReader().read(cfg_dir / "baltic_all-parameters.csv")
    params = json.loads(Path(args.results_json).read_text())["parameters"]
    for key, val in params.items():
        got = cfg.get(key.lower())
        assert got is not None and abs(float(got) - float(val)) < 1e-6, f"{key}: {got!r} != {val}"
    for i in range(8):  # also verify the shepherd-type writes (not in params)
        got = cfg.get(f"stock.recruitment.type.sp{i}")
        assert got == "shepherd", f"stock.recruitment.type.sp{i}: {got!r} != 'shepherd'"
    print(f"applied {len(params)} params + set 8x shepherd type; roundtrip OK")


if __name__ == "__main__":
    main()
