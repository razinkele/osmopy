"""Convert a bundled OSMOSE config to native 4.4.0 IN PLACE (C1 Task 3/4).

Rewrites ONLY the key-value parameter files (those reachable via ``osmose.configuration.*``
includes) — never the matrix/map data CSVs (predation-accessibility, maps, masks). Per param
file: rename keys to 4.4.0 spelling (RENAMES_440, longest-prefix-first, skip-if-exists; the lossy
ingestion-bioen merge is left for the reader), scale the additional-larval-mortality rate to
rate/year (x ndt), stamp osmose.version. KEEPS species.lmax / species.beta (the Python engine
reads them; the Java write path drops them). Does NOT bake the Java-only resource-forcing keys
into source — write_temp_config emits those at Java-stage time (the Python engine ignores them).

  python scripts/migrate_bundled_to_440.py data/eec_full
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from osmose.config.aliases import (
    RENAMES_440,
    _LARVA_RATE_RE,
    _ndtperyear,
    _numeric_version,
)
from osmose.config.reader import OsmoseConfigReader
from osmose.demo import _version_tuple

ROOT = Path(__file__).resolve().parents[1]
IN_SCOPE = {"eec_full", "minimal", "baltic", "baltic_ev", "examples"}
TARGET = "4.4.1"
_SEP_RE = re.compile(r"\s*[=;,:\t]\s*")


# The ingestion-bioen rename is a LOSSY MERGE (many bioen keys -> one base key) whose bioen value the
# engine reads for a bioenergetics config (config.py:1796). Renaming it in source would either drop a
# needed value or collide into a duplicate, so leave it as-is — the reader canonicalizes it on read
# with the correct merge semantics, and parity holds.
_NO_RENAME = {"predation.ingestion.rate.max.bioen"}


def _rename_forward(key: str) -> str:
    """Longest-prefix-first OLD->NEW rename (mirror of to_target_keys' inverse loop)."""
    for old in sorted(RENAMES_440, key=len, reverse=True):
        if old in _NO_RENAME:
            continue
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


def _convert_line(line: str, ndt: float, original_keys: set[str]) -> str | None:
    """Convert one config line to native 4.4.0; return None to DROP it (skip-if-exists rename)."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line
    m = _SEP_RE.search(line)
    if not m:
        return line
    sep, key, value = m.group(0), line[: m.start()].strip(), line[m.end() :].rstrip("\n")
    new_key = _rename_forward(key.lower())
    # skip-if-exists (mirror migrate_config): if this is a rename (OLD->NEW) and the NEW key already
    # exists in the source, DROP the OLD line so the rename doesn't collide into a duplicate key.
    if new_key != key.lower() and new_key in original_keys:
        return None
    if new_key == "osmose.version":
        value = TARGET
    elif _LARVA_RATE_RE.match(new_key) and ndt:
        try:
            value = repr(float(value) * ndt)  # per-cohort -> rate/year, full precision
        except ValueError:
            pass
    return f"{new_key}{sep}{value}\n"


_BOB_RESOURCE_SP = range(8, 14)  # sp8..sp13 resources
_FORCING_24 = "ltl/roms_n2p2z2d2_biscay_24step.nc"


def _convert_bob_native(config_dir: Path) -> None:
    """BoB-specific fully-native fixups (run AFTER the generic per-line conversion).

    (a) add per-species species.file.spN -> the 24-step forcing (drives both the Python
        species.type forcing read AND the Java-stage species.biomass.file emit); (b) drop every
        ltl.* key across all param files (a single leftover ltl.name.rscN re-routes the Python
        engine back onto _load_config_ltl). We KEEP species.tl.spN unchanged — native EEC does the
        same, the 4.4.1 Java jar reads species.tl (ResourceSpecies.java), and the Python
        species.type path simply defaults resource TL (diagnostic-only; EEC does this and parity
        passed). Do NOT rename species.tl -> species.trophic.level (that would break the Java read).
    """
    master = next(iter(config_dir.glob("*all-parameters*.csv")))
    for f in _collect_param_files(master):
        out = []
        for ln in f.read_text().splitlines(keepends=True):
            s = ln.strip()
            if s and not s.startswith("#"):
                m = _SEP_RE.search(ln)
                if m:
                    key = ln[: m.start()].strip().lower()
                    if key.startswith("ltl."):
                        continue  # drop the whole ltl.* family
            out.append(ln)
        f.write_text("".join(out))
    # append the per-species forcing paths to the master (idempotent: skip if present)
    text = master.read_text()
    existing = {ln.split(_SEP_RE.search(ln).group(0))[0].strip().lower()
                for ln in text.splitlines() if _SEP_RE.search(ln) and not ln.strip().startswith("#")}
    add = [f"species.file.sp{i} ; {_FORCING_24}\n"
           for i in _BOB_RESOURCE_SP if f"species.file.sp{i}" not in existing]
    if add:
        if not text.endswith("\n"):
            text += "\n"
        master.write_text(text + "# Osmose 4.4.1 - per-species resource forcing (24-step)\n" + "".join(add))


def _original_keys(param_files: list[Path]) -> set[str]:
    """All lowercased keys present across the source param files (pre-rename)."""
    keys: set[str] = set()
    for f in param_files:
        for line in f.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                m = _SEP_RE.search(line)
                if m:
                    keys.add(line[: m.start()].strip().lower())
    return keys


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
    original_keys = _original_keys(param_files)  # for skip-if-exists rename collisions
    for f in param_files:
        out_lines = [
            _convert_line(ln, ndt, original_keys) for ln in f.read_text().splitlines(keepends=True)
        ]
        f.write_text("".join(ln for ln in out_lines if ln is not None))
    if name == "examples":
        _convert_bob_native(config_dir)
    # NOTE: the Java-only NETCDF_BIOMASS resource-forcing keys (species.biomass.{file,mode,varname})
    # are NOT baked into source — they are a write-for-Java concern that write_temp_config ->
    # to_target_keys emits at stage time, and the Python engine does not read them (H3). Baking them
    # in would make the Python validator flag them as unknown keys.
    print(f"{name}: converted {len(param_files)} param file(s) -> native 4.4.0")


if __name__ == "__main__":
    convert_config(Path(sys.argv[1]).resolve())
