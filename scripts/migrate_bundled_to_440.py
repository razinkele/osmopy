"""Convert a bundled OSMOSE config to native 4.4.0 IN PLACE (C1 Task 3/4).

Rewrites ONLY the key-value parameter files (those reachable via ``osmose.configuration.*``
includes) — never the matrix/map data CSVs (predation-accessibility, maps, masks). Per param
file: rename keys to 4.4.0 spelling (RENAMES_440, longest-prefix-first), scale the
additional-larval-mortality rate to rate/year (x ndt), stamp osmose.version. KEEPS species.lmax /
species.beta (the Python engine reads them; the Java write path drops them). Emits the 4.4.x
resource-forcing keys into the master (inert for the Python engine; needed by the 4.4.1 jar).

  python scripts/migrate_bundled_to_440.py data/eec_full
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from osmose.config.aliases import (
    RENAMES_440,
    _emit_resource_biomass_forcing,
    _LARVA_RATE_RE,
    _ndtperyear,
    _numeric_version,
)
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import _version_tuple

ROOT = Path(__file__).resolve().parents[1]
IN_SCOPE = {"eec_full", "minimal", "baltic", "baltic_ev"}
TARGET = "4.4.1"
_SEP_RE = re.compile(r"\s*[=;,:\t]\s*")


def _rename_forward(key: str) -> str:
    """Longest-prefix-first OLD->NEW rename (mirror of to_target_keys' inverse loop)."""
    for old in sorted(RENAMES_440, key=len, reverse=True):
        if key == old or key.startswith(old + "."):
            return RENAMES_440[old] + key[len(old) :]
    return key


def _collect_param_files(master: Path, seen: set[Path] | None = None) -> list[Path]:
    """Master + all key-value param files reachable via osmose.configuration.* includes."""
    seen = seen if seen is not None else set()
    master = master.resolve()
    if master in seen or not master.exists():
        return []
    seen.add(master)
    files = [master]
    for line in master.read_text().splitlines():
        if line.strip().lower().startswith("osmose.configuration."):
            m = _SEP_RE.search(line)
            if m:
                files += _collect_param_files(master.parent / line[m.end() :].strip(), seen)
    return files


def _raw_version(master: Path) -> str:
    """The RAW osmose.version from the file (the reader canonicalizes it to 4.4.0 in memory)."""
    for line in master.read_text().splitlines():
        if line.strip().lower().startswith("osmose.version"):
            m = _SEP_RE.search(line)
            if m:
                return line[m.end() :].strip()
    return "4.3.3"


def _convert_line(line: str, ndt: float) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line
    m = _SEP_RE.search(line)
    if not m:
        return line
    sep, key, value = m.group(0), line[: m.start()].strip(), line[m.end() :].rstrip("\n")
    new_key = _rename_forward(key.lower())
    if new_key == "osmose.version":
        value = TARGET
    elif _LARVA_RATE_RE.match(new_key) and ndt:
        try:
            value = repr(float(value) * ndt)  # per-cohort -> rate/year, full precision
        except ValueError:
            pass
    return f"{new_key}{sep}{value}\n"


def convert_config(config_dir: Path) -> None:
    name = config_dir.name
    if name not in IN_SCOPE:
        raise SystemExit(f"{name} not in scope {IN_SCOPE} (BoB/examples excluded)")
    master = next(iter(config_dir.glob("*all-parameters*.csv")))
    if _numeric_version(_raw_version(master)) >= _version_tuple("4.4.0"):
        print(f"{name}: already >= 4.4.0, skipping")
        return
    merged = dict(OsmoseConfigReader().read(str(master)))
    ndt = _ndtperyear(merged) or 1.0
    param_files = _collect_param_files(master)
    for f in param_files:
        f.write_text(
            "".join(_convert_line(ln, ndt) for ln in f.read_text().splitlines(keepends=True))
        )
    # resource-forcing keys: inert for the Python engine (H3), needed by the 4.4.1 jar; append the
    # NEW ones to the master (species.biomass.* names are version-stable).
    emitted = _emit_resource_biomass_forcing(dict(merged))
    new_keys = {k: v for k, v in emitted.items() if k not in merged}
    if new_keys:
        extra = "".join(f"{k} ; {v}\n" for k, v in sorted(new_keys.items()))
        master.write_text(master.read_text() + "\n# --- 4.4.x resource forcing (C1) ---\n" + extra)
    print(
        f"{name}: converted {len(param_files)} param file(s) -> native 4.4.0 "
        f"({len(new_keys)} resource-forcing keys emitted)"
    )


if __name__ == "__main__":
    convert_config(Path(sys.argv[1]).resolve())
