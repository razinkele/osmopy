"""Stage Baltic for OSMOSE 4.4.1 (background-species adaptation) and validate it loads + runs.

The staging recipe lives in ``osmose.java_background_staging`` (shared with the UI Java run path);
this script stages Baltic, runs the 4.4.1 jar, and asserts it loads + the predators feed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from osmose.java_background_staging import stage_background_for_java

ROOT = Path("/home/razinka/osmose/osmose-python")
BALTIC = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
JAR = ROOT / "osmose-java" / "osmose-4.4.1-jar-with-dependencies.jar"


def assert_predators_feed(out_dir: Path, control_out_dir: Path | None = None) -> None:
    """Assert that background predators (GreySeal/Cormorant) exert predation on focal species.

    Primary: check diet matrix CSV for non-zero predation by GreySeal/Cormorant columns.
    Fallback: compare focal biomass between with-background and no-background runs (>5% difference
    for at least one species proves predation impact).

    Raises AssertionError if feeding cannot be confirmed.
    """
    import csv as csv_mod

    # Primary check: diet matrix output (comma-separated in 4.4.1)
    diet_files = list(out_dir.rglob("*dietMatrix*")) + list(out_dir.rglob("*predatorPressure*"))
    found_feeding = False
    for f in diet_files:
        try:
            text = f.read_text(errors="replace")
            if "GreySeal" not in text and "Cormorant" not in text:
                continue
            lines = text.splitlines()
            # Skip first comment line; second line is header
            lines = [ln for ln in lines if not ln.startswith('"% of')]
            if len(lines) < 2:
                continue
            header = next(csv_mod.reader([lines[0]]))
            gs_cols = [i for i, h in enumerate(header) if "GreySeal" in h or "Cormorant" in h]
            if not gs_cols:
                continue
            nonzero_count = 0
            for row in csv_mod.reader(lines[1:]):
                for ci in gs_cols:
                    if ci < len(row):
                        v = row[ci].strip().strip('"')
                        if v and v.lower() != "nan":
                            try:
                                if float(v) > 0:
                                    nonzero_count += 1
                            except ValueError:
                                pass
            if nonzero_count > 0:
                found_feeding = True
                print(f"Feeding confirmed in: {f.name} ({nonzero_count} non-zero diet values)")
                break
        except Exception as exc:
            print(f"WARNING: Could not parse {f.name}: {exc}")

    if found_feeding:
        print("PASS: GreySeal/Cormorant feeding confirmed via diet output.")
        return

    # Fallback: compare with-background vs control (nbackground=0) run
    if control_out_dir is not None:
        bg_biomass = _read_biomass_means(out_dir)
        ctrl_biomass = _read_biomass_means(control_out_dir)
        if bg_biomass and ctrl_biomass:
            diffs = []
            for sp in bg_biomass:
                if sp in ctrl_biomass and ctrl_biomass[sp] > 0:
                    pct = abs(bg_biomass[sp] - ctrl_biomass[sp]) / ctrl_biomass[sp]
                    diffs.append((sp, pct))
            max_diff = max((d for _, d in diffs), default=0.0)
            print(f"Max focal biomass diff vs control: {max_diff:.1%}")
            if max_diff >= 0.05:
                print("PASS: Background predators exert measurable predation (>5% biomass impact).")
                return
            else:
                raise AssertionError(
                    f"FAIL: Max biomass diff {max_diff:.1%} < 5% threshold. "
                    "Background predators not confirming significant feeding. Check BG_ACCESS values."
                )
        raise AssertionError(
            "FAIL: Could not compare biomass with control run (missing or incomplete output)."
        )

    # Neither check was conclusive
    raise AssertionError(
        f"FAIL: Could not confirm GreySeal/Cormorant feeding from diet output. "
        f"Diet files found: {[f.name for f in diet_files[:5]]}"
    )


def _read_biomass_means(out_dir: Path) -> dict[str, float]:
    """Read mean focal-species biomass from output CSVs. Returns {species: mean_biomass}.

    OSMOSE 4.4.1 uses comma-separated output; the first line is a comment.
    """
    import csv as csv_mod

    result: dict[str, float] = {}
    files_failed = 0
    for f in out_dir.rglob("*biomass*.csv"):
        try:
            lines = [ln for ln in f.read_text().splitlines() if not ln.startswith('"Mean')]
            if len(lines) < 2:
                continue
            header = next(csv_mod.reader([lines[0]]))
            cols: dict[str, list[float]] = {}
            for ln in lines[1:]:
                row = next(csv_mod.reader([ln]))
                for col_idx, col in enumerate(header[1:], start=1):
                    sp = col.strip().strip('"')
                    if not sp or col_idx >= len(row):
                        continue
                    v = row[col_idx].strip().strip('"')
                    try:
                        cols.setdefault(sp, []).append(float(v))
                    except ValueError:
                        pass
            for sp, vals in cols.items():
                if vals:
                    result[sp] = float(np.mean(vals))
        except Exception as exc:
            print(f"WARNING: Failed to parse biomass file {f.name}: {exc}")
            files_failed += 1
    if files_failed > 0:
        print(f"Note: {files_failed} biomass file(s) failed to parse.")
    return result


def stage_and_run(years: int = 3, out: Path | None = None) -> int:
    import tempfile

    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.run import write_temp_config

    raw = dict(OsmoseConfigReader().read(str(BALTIC)))
    tmp = Path(out or tempfile.mkdtemp(prefix="baltic440_"))
    stage = tmp / "stage"
    write_temp_config(raw, stage, source_dir=BALTIC.parent, target_version="4.4.1")
    master = stage / "osm_all-parameters.csv"
    overrides = stage_background_for_java(stage, raw)  # shared staging recipe (osmose module)
    odir = tmp / "out"
    odir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Xmx2g",
        "-jar",
        str(JAR),
        str(master),
        f"-Poutput.dir.path={odir}",
        f"-Psimulation.time.nyear={years}",
        "-Poutput.start.year=0",
        *[f"-P{k}={v}" for k, v in overrides.items()],  # incl. output.cutoff.enabled=false
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print("exit:", r.returncode)
    if r.returncode != 0:
        print("\n".join((r.stdout or "").splitlines()[-12:]))
        print("\n".join((r.stderr or "").splitlines()[-12:]))
    print("output CSVs:", len(list(odir.rglob("*.csv"))))

    if r.returncode == 0:
        # Acceptance assertions
        biomass = _read_biomass_means(odir)
        if biomass:
            collapsed = [sp for sp, v in biomass.items() if v == 0.0]
            if collapsed:
                raise AssertionError(f"FAIL: Focal species collapsed to zero: {collapsed}")
            else:
                print(f"PASS: All {len(biomass)} focal species have non-zero mean biomass.")
        # Feeding check (primary: diet files; fallback: comparison run)
        assert_predators_feed(odir)

    return r.returncode


if __name__ == "__main__":
    try:
        exit_code = stage_and_run()
        sys.exit(0 if exit_code == 0 else 1)
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
