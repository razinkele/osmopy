"""Reconcile a STAGED OSMOSE config so the Java 4.4.1 engine can load and run it.

Java 4.4.1 mangles species names: ``Species.java`` builds the internal name via
``species.name.spN.replaceAll("_","").replaceAll("-","")`` (``cod_west`` -> ``codwest``),
but leaves every NAME-BASED reference untouched — ``movement.species.mapN`` values and the
predation-accessibility / fishery-matrix headers and rows still carry the underscore. So
``MapSet.getSpecies`` and ``Matrix.getIndexPrey/getIndexPred`` can no longer resolve them and
the run aborts (``"... does not match any predefined species name"``). The Python engine keeps
the raw names, so this only bites the Java path.

This pass rewrites a staged config to what Java expects, mirroring Java's own stripping and
repairing the matrix inconsistencies the disaggregation left behind (which Python tolerates):

  1. **Name sanitize** — strip ``_``/``-`` from species + fishery names in the name-resolving
     master keys (``species.name.sp*``, ``movement.species.map*``, ``fisheries.name.fsh*``) and
     in the matrix headers/rows, so references match Java's stripped internal names.
  2. **Column dedup** — collapse duplicate predator columns in predation-accessibility (a config
     that already ships a background-predator column, then has it re-added by the background
     staging, ends up with two).
  3. **Fishery-matrix reconcile** — give catchability + discards the same fishery columns and a
     row for every prey in the accessibility universe, preserving existing cell values by name
     and zero-filling the rest (the disaggregation updated catchability but left discards stale
     with an aggregate ``cod`` row and no ``trawlcod_east`` column). Zero-fill is faithful to the
     Python engine, which silently applies a zero discard rate to any species absent from the
     discards file.

Staged-copy only — operates in place on ``stage_dir``; never reads or writes ``data/``. General,
not Baltic-specific: a no-op on a config that already has alphanumeric names and consistent
matrices. Intended to run AFTER ``write_temp_config`` + ``stage_background_for_java``.
"""

from __future__ import annotations

from pathlib import Path

# Master parameter keys whose VALUE is a species or fishery name Java resolves against the
# (stripped) internal name. File-path keys (movement.file.map*) are deliberately excluded — a
# path is not a name and mangling it would break the file lookup.
_NAME_VALUE_KEYS = ("species.name.sp", "movement.species.map", "fisheries.name.fsh")


def sanitize_java_name(name: str) -> str:
    """Return *name* as Java 4.4.1 stores it internally: ``_`` and ``-`` removed."""
    return name.replace("_", "").replace("-", "")


def _read_matrix(path: Path, sep: str) -> list[list[str]]:
    return [ln.split(sep) for ln in path.read_text().splitlines() if ln.strip()]


def _write_matrix(path: Path, rows: list[list[str]], sep: str) -> None:
    path.write_text("\n".join(sep.join(cell for cell in row) for row in rows) + "\n")


def _sanitize_master(master: Path) -> None:
    """Strip ``_``/``-`` from the VALUE of every name-resolving key; assert names stay unique."""
    out: list[str] = []
    stripped_names: dict[str, list[str]] = {}
    for line in master.read_text().splitlines():
        if ";" in line:
            key, _, value = line.partition(";")
            if key.strip().startswith(_NAME_VALUE_KEYS):
                new_value = " " + sanitize_java_name(value.strip())
                out.append(f"{key};{new_value}")
                if key.strip().startswith("species.name.sp"):
                    stripped_names.setdefault(sanitize_java_name(value.strip()), []).append(
                        value.strip()
                    )
                continue
        out.append(line)
    collisions = {k: v for k, v in stripped_names.items() if len(v) > 1}
    if collisions:
        raise ValueError(
            f"species names collide after stripping '_'/'-' for Java: {collisions}. "
            "Rename so the alphanumeric-only forms stay distinct."
        )
    master.write_text("\n".join(out) + "\n")


def _reconcile_accessibility(path: Path) -> list[str]:
    """Sanitize names + drop duplicate predator columns; return the prey-row universe."""
    rows = [[sanitize_java_name(c) for c in r] for r in _read_matrix(path, ";")]
    header = rows[0]
    seen: set[str] = set()
    keep: list[int] = []
    for i, name in enumerate(header):
        if i == 0 or name not in seen:  # col 0 is the "v Prey / Predator >" label
            keep.append(i)
            seen.add(name)
    rows = [[r[i] if i < len(r) else "0" for i in keep] for r in rows]
    _write_matrix(path, rows, ";")
    return [r[0] for r in rows[1:]]


def _reconcile_fishery_matrix(
    path: Path, fisheries_header: list[str], prey_universe: list[str], sep: str = ","
) -> None:
    """Rewrite *path* to have columns *fisheries_header* and one row per prey in
    *prey_universe*, preserving existing (prey, fishery) cell values by name (both sanitized)
    and zero-filling the rest."""
    old = _read_matrix(path, sep)
    old_header = [sanitize_java_name(c) for c in old[0]]
    lookup: dict[tuple[str, str], str] = {}
    for row in old[1:]:
        prey = sanitize_java_name(row[0])
        for j in range(1, len(row)):
            lookup[(prey, old_header[j])] = row[j]
    new = [list(fisheries_header)]
    for prey in prey_universe:
        new.append([prey] + [lookup.get((prey, f), "0") for f in fisheries_header[1:]])
    _write_matrix(path, new, sep)


def _resolve(stage_dir: Path, master_cfg: dict[str, str], key: str, default: str) -> Path | None:
    """Resolve a staged matrix path from a master file-key (falls back to the standard name)."""
    rel = master_cfg.get(key, default).strip()
    p = stage_dir / rel
    return p if p.exists() else None


def reconcile_config_for_java(
    stage_dir: Path, master_name: str = "osm_all-parameters.csv"
) -> dict[str, int]:
    """Make the staged config at *stage_dir* loadable/runnable by Java 4.4.1 (see module docstring).

    In-place on the staged copy. Returns a small summary (counts) for logging/tests.
    """
    stage_dir = Path(stage_dir)
    master = stage_dir / master_name
    # cheap key->value map of the master (last value wins), for resolving matrix file paths
    cfg: dict[str, str] = {}
    for line in master.read_text().splitlines():
        if ";" in line:
            k, _, v = line.partition(";")
            cfg[k.strip()] = v.strip()

    _sanitize_master(master)

    summary = {"accessibility": 0, "catchability": 0, "discards": 0}
    acc_path = _resolve(
        stage_dir, cfg, "predation.accessibility.file", "predation-accessibility.csv"
    )
    prey_universe: list[str] = []
    if acc_path is not None:
        prey_universe = _reconcile_accessibility(acc_path)
        summary["accessibility"] = 1

    cat_path = _resolve(stage_dir, cfg, "fisheries.catchability.file", "fishery-catchability.csv")
    if cat_path is not None and prey_universe:
        fisheries_header = [sanitize_java_name(c) for c in _read_matrix(cat_path, ",")[0]]
        _reconcile_fishery_matrix(cat_path, fisheries_header, prey_universe)
        summary["catchability"] = 1
        disc_path = _resolve(stage_dir, cfg, "fisheries.discards.file", "fishery-discards.csv")
        if disc_path is not None:
            _reconcile_fishery_matrix(disc_path, fisheries_header, prey_universe)
            summary["discards"] = 1
    return summary
