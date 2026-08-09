"""SP-A certification — 50 yr × multi-seed persistence/in-envelope table for a candidate config.

Given a parameter set (from the ε-sweep JSON, or "current" for the un-recalibrated baseline), runs the
Python engine 50 yr × 5 seeds and reports, per focal species: min biomass, final-decade mean, whether
it persists (min > 0.1·ICES-lower) and stays in-envelope (lower ≤ final-decade mean ≤ upper). With
--java, a single Java 4.4.1 cross-check (via the C2 background staging) compares survivor sets.

    PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current
    PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params <sweep.json> --java

Baseline sanity: `--params current` must reproduce the known collapse (few survivors), proving the
harness actually detects instability.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from osmose.engine import PythonEngine

_JAR = Path(
    os.environ.get(
        "OSMOSE_JAR",
        str(
            Path(__file__).resolve().parents[1]
            / "osmose-java"
            / "osmose-4.4.1-jar-with-dependencies.jar"
        ),
    )
)

# Envelopes (data/baltic/reference/biomass_targets.csv): (lower, upper) tonnes.
# Kept literal so every historical certification stays byte-comparable; _load_target_weights()
# below re-reads the source file and RAISES if the two ever diverge.
ENVELOPE = {
    "cod_west": (4000, 25000),
    "cod_east": (60000, 85000),
    "herring": (800000, 3000000),
    "sprat": (800000, 2500000),
    "flounder": (20000, 100000),
    "perch": (8000, 50000),
    "pikeperch": (4000, 25000),
    "smelt": (20000, 120000),
    "stickleback": (50000, 500000),
}
FOCAL = list(ENVELOPE)

_TARGETS_CSV = (
    Path(__file__).resolve().parents[1] / "data" / "baltic" / "reference" / "biomass_targets.csv"
)

# Confidence tiers, from the `weight` column of biomass_targets.csv, whose header defines
# 1.0 = high (well-assessed), 0.5 = medium, 0.2 = low (poorly resolved at grid scale).
# Species at or below this threshold are reported as INDICATIVE and excluded from the headline
# verdict: ICES does not assess Baltic pikeperch, perch, smelt or stickleback at all, and the file
# sources them as "Literature estimate for coastal Baltic" with the note "Concentrated in
# estuaries/lagoons; coarse grid under-resolves". Scoring those pass/fail alongside category-1
# analytical assessments was making the headline number a statement about the weakest targets
# rather than about the model. (2026-08-04)
INDICATIVE_MAX_WEIGHT = 0.3


def _load_target_weights() -> dict[str, float]:
    """Read per-species confidence weights, and verify the envelopes still match ENVELOPE.

    Only ``ssb``/``biomass`` rows are stock targets; the file also carries a parallel set of
    ``catch`` rows (a landings-based fallback with its own weights and much smaller bounds), and
    reading those by mistake would silently swap in the wrong envelope for five species.
    """
    import csv

    weights: dict[str, float] = {}
    with open(_TARGETS_CSV) as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            if row["reference_point_type"] not in ("ssb", "biomass"):
                continue
            sp = row["species"]
            if sp not in ENVELOPE:
                continue
            lo, hi = float(row["lower_tonnes"]), float(row["upper_tonnes"])
            if (lo, hi) != tuple(float(x) for x in ENVELOPE[sp]):
                raise ValueError(
                    f"{_TARGETS_CSV.name} envelope for {sp!r} is ({lo:g}, {hi:g}) but ENVELOPE has "
                    f"{ENVELOPE[sp]}. Reconcile before certifying — a silent divergence here would "
                    f"invalidate comparison against every prior certification."
                )
            weights[sp] = float(row["weight"])
    missing = set(ENVELOPE) - set(weights)
    if missing:
        raise ValueError(f"No ssb/biomass target row in {_TARGETS_CSV.name} for: {sorted(missing)}")
    return weights


TARGET_WEIGHT = _load_target_weights()
ASSESSED = [sp for sp in FOCAL if TARGET_WEIGHT[sp] > INDICATIVE_MAX_WEIGHT]
INDICATIVE = [sp for sp in FOCAL if TARGET_WEIGHT[sp] <= INDICATIVE_MAX_WEIGHT]
CERT_SEEDS = (42, 123, 7, 999, 2024)

# Python-only features with no Java-jar equivalent: the Java cross-check arm runs with these
# pinned off and SAYS so, keeping the arm runnable while labelling the divergence
# (spec 2026-08-08 §4 Phase 1; runner.java_engine_block_reason blocks Java otherwise).
# ltl.oxygen.benthos.enabled (spec Phase 2a, adopted 2026-08-09): the O2->benthos
# carrying-capacity coupling is Python-only. Java DOES read the oxygen.* forcing keys
# (bioenergetics f_o2), so those are NOT pinned here — only the coupling flag is.
JAVA_INCOMPATIBLE_PINS = {
    "ltl.depletable.enabled": "false",
    "ltl.oxygen.benthos.enabled": "false",
}


def pin_java_incompatible(cfg: dict) -> tuple[dict, list[str]]:
    """Copy of ``cfg`` with Python-only flags forced off, plus the list of keys pinned."""
    out = dict(cfg)
    pinned = []
    for key, off_value in JAVA_INCOMPATIBLE_PINS.items():
        if str(out.get(key, "")).strip().lower() == "true":
            out[key] = off_value
            pinned.append(key)
    return out, pinned


def _prepare_java_cfg(params: dict[str, str]) -> tuple[dict, list[str]]:
    """Demo config + params + Java pinning — the exact config certify_java stages (test seam)."""
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg.update(params)  # bake the recalibrated params into the config
    cfg, pinned = pin_java_incompatible(cfg)
    if pinned:
        print(f"(Java arm: pinned off Python-only features: {', '.join(pinned)})")
    return cfg, pinned


def _load_params(source: str) -> dict[str, str]:
    """'current' -> no overrides; otherwise a sweep JSON path -> its best in-envelope point's params."""
    if source == "current":
        return {}
    data = json.loads(Path(source).read_text())
    # pick the tightest-epsilon point that recorded params (front is loose->tight)
    candidates = [p for p in data if p.get("params")]
    if not candidates:
        raise SystemExit(f"no front point with params in {source}")
    best = candidates[-1]
    return {k: str(v) for k, v in best["params"].items()}


def _species_row(bio, sp: str) -> dict:
    lo, hi = ENVELOPE[sp]
    v = np.asarray(bio[sp].values, float) if sp in bio.columns else np.array([0.0])
    late = v[-10:] if len(v) >= 10 else v
    # `persists` describes the EQUILIBRIUM, so the minimum is scoped to the final decade like
    # `in_envelope`'s mean. Using the whole-run minimum made it a statement about the SEEDING
    # TRANSIENT: the 2026-08-01 seeding A/B scored two arms 2/9 vs 6/9 whose final-decade means were
    # within +-5%, purely because one seeded more eggs and dipped less deeply during bootstrap
    # (docs/baltic_seeding_ab_comparison_2026-08-01.md). cod_east dipping to 17 t before settling at
    # ~83 kt inside envelope was reported as a collapse. A genuine LATE collapse still fails, because
    # it happens within the final decade.
    vmin, late_mean = float(late.min()), float(np.mean(late))
    persists = vmin > 0.1 * lo
    in_env = lo <= late_mean <= hi
    return {"min": vmin, "late_mean": late_mean, "persists": persists, "in_envelope": in_env}


def certify_python(
    params: dict[str, str], n_years: int, seeds, seeding_mode: str | None = None
) -> dict:
    """Run Python n_years x seeds; aggregate per-species persist/in-envelope (worst-case across seeds).

    ``seeding_mode`` (GitHub #143) must match the mode the params were CALIBRATED under — certifying
    linear-refitted parameters under stock_recruitment seeding would score them against dynamics they
    were never fitted for. Applied after ``params`` so it cannot be silently overridden by a stale key
    carried in a results JSON.
    """
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    cfg = dict(OsmoseConfigReader().read(str(res["config_file"])))
    cfg["simulation.time.nyear"] = str(n_years)
    cfg.update(params)
    if seeding_mode is not None:
        cfg["population.seeding.mode"] = seeding_mode
        print(f"Seeding mode: {seeding_mode}")
    per_seed = []
    for seed in seeds:
        bio = PythonEngine().run_in_memory(cfg, seed=seed).biomass()
        per_seed.append({sp: _species_row(bio, sp) for sp in FOCAL})
    # worst-case: a species passes only if it persists AND is in-envelope in EVERY seed
    table = {}
    for sp in FOCAL:
        rows = [ps[sp] for ps in per_seed]
        table[sp] = {
            "persists": all(r["persists"] for r in rows),
            "in_envelope": all(r["in_envelope"] for r in rows),
            "min_biomass": min(r["min"] for r in rows),
            "late_mean_range": [
                min(r["late_mean"] for r in rows),
                max(r["late_mean"] for r in rows),
            ],
        }
    return table


def certify_java(params: dict[str, str], n_years: int) -> dict | None:
    """Single Java 4.4.1 run (staged via the C2 background recipe) -> per-species table, or None if
    the jar is missing / the run fails. A coarse cross-engine consistency check (Baltic is not
    bit-equal cross-engine).

    Deliberately takes no seed: Java 4.4.1 exposes only a ``simulation.fixedseed.enabled``
    toggle, not a numeric seed, so there is no way to run it "at seed 42". This function
    previously accepted a ``seed`` argument and silently ignored it, which is where the
    "single seed 42" claim in the 2026-07-29/30 cross-check notes came from.
    """
    if not _JAR.exists():
        print(f"(Java cross-check skipped: jar not found at {_JAR})")
        return None
    from ui.pages.run import stage_config_for_java

    cfg, _pinned = _prepare_java_cfg(params)
    tmp = Path(tempfile.mkdtemp())
    res = osmose_demo("baltic", tmp)
    stage = tmp / "stage"
    # Writes the master, stages background species (overrides incl. output.cutoff.enabled=false),
    # and reconciles names/matrices so Java can resolve cod_west/cod_east. Shared with the Run
    # tab so the two staging paths cannot drift (GitHub #138).
    master, overrides = stage_config_for_java(
        cfg, stage, res["config_file"].parent, target_version="4.4.1"
    )
    odir = tmp / "out"
    odir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Xmx2g",
        "-jar",
        str(_JAR),
        str(master),
        f"-Poutput.dir.path={odir}",
        f"-Psimulation.time.nyear={n_years}",
        "-Poutput.start.year=0",
        *[f"-P{k}={v}" for k, v in overrides.items()],
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"(Java run failed, exit {r.returncode}):")
        print("\n".join((r.stdout or "").splitlines()[-8:]))
        return None
    # Read the 4.4.1 Java biomass CSV directly (comma-separated, first line a "Mean..." comment,
    # species NAMES as columns). OsmoseResults does not map the Java biomass to focal-named columns.
    import csv as csv_mod

    series: dict[str, list[float]] = {}
    for f in sorted(odir.rglob("*biomass_Simu*.csv")):
        rows = [ln for ln in f.read_text().splitlines() if not ln.lstrip().startswith('"Mean')]
        if len(rows) < 2:
            continue
        header = next(csv_mod.reader([rows[0]]))
        for ln in rows[1:]:
            cells = next(csv_mod.reader([ln]))
            for ci, col in enumerate(header[1:], start=1):
                sp = col.strip().strip('"')
                if sp and ci < len(cells):
                    try:
                        series.setdefault(sp, []).append(float(cells[ci].strip().strip('"')))
                    except ValueError:
                        pass
    if not series:
        print("(Java biomass output not found or unparseable)")
        return None

    return java_table_from_series(series)


def java_table_from_series(series: dict[str, list[float]]) -> dict:
    """Build the per-species certification table from Java's biomass series.

    Java writes every column header as the SANITIZED species name — ``Species.java`` strips
    ``_``/``-``, and ``reconcile_config_for_java`` mirrors that into the staged config — so a
    focal species is resolved through ``sanitize_java_name`` (identity for names that were
    already alphanumeric).

    A missing or empty column is a harness bug, never an extinction. Substituting 0.0 here is
    what reported both cod stocks extinct in the 2026-07-29 and 2026-07-30 cross-checks while
    Java in fact had them alive, so this raises instead.
    """
    from osmose.java_config_reconcile import sanitize_java_name

    table: dict[str, dict] = {}
    missing: list[str] = []
    for sp in FOCAL:
        col = sanitize_java_name(sp)
        values = series.get(col)
        if not values:
            missing.append(f"{sp} (expected column {col!r})")
            continue
        lo, hi = ENVELOPE[sp]
        v = np.asarray(values, float)
        late = v[-10:] if len(v) >= 10 else v
        # Final-decade minimum, matching _species_row — see the note there.
        vmin, late_mean = float(late.min()), float(np.mean(late))
        table[sp] = {
            "persists": vmin > 0.1 * lo,
            "in_envelope": lo <= late_mean <= hi,
            "min_biomass": vmin,
            "late_mean_range": [late_mean, late_mean],
        }
    if missing:
        raise KeyError(
            "Java biomass output has no usable column for: "
            + "; ".join(missing)
            + f". Columns present: {sorted(series)}"
        )
    return table


def _print_table(engine: str, table: dict) -> int:
    """Print the per-species table. Returns the ASSESSED-tier pass count (the headline)."""
    print(f"\n=== {engine}: per-species certification ===")
    counts = {}
    for tier, members in (("ASSESSED", ASSESSED), ("INDICATIVE", INDICATIVE)):
        if not members:
            continue
        print(
            f"  -- {tier} (weight {'>' if tier == 'ASSESSED' else '<='} {INDICATIVE_MAX_WEIGHT}) --"
        )
        ok = 0
        for sp in members:
            t = table[sp]
            good = t["persists"] and t["in_envelope"]
            ok += good
            flag = "PASS" if good else ("persists" if t["persists"] else "COLLAPSE")
            print(
                f"  {sp:12s} w={TARGET_WEIGHT[sp]:<4g} {flag:9s} "
                f"min={t['min_biomass']:.2e} late_mean={t['late_mean_range']}"
            )
        counts[tier] = ok
    a, i = counts.get("ASSESSED", 0), counts.get("INDICATIVE", 0)
    print(f"  --> ASSESSED {a}/{len(ASSESSED)} persistent & in-envelope   (headline)")
    if INDICATIVE:
        print(
            f"      indicative {i}/{len(INDICATIVE)} — low-confidence targets, not part of the verdict"
        )
        print(
            f"      all species {a + i}/{len(FOCAL)} (legacy figure, for comparison with pre-2026-08-04 notes)"
        )
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default="current", help="'current' or a stability_sweep.json path")
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CERT_SEEDS))
    ap.add_argument(
        "--seeding-mode",
        choices=["stock_recruitment", "linear"],
        default=None,
        help="Python seeding mode (GitHub #143); must match what the params were calibrated under. "
        "Java always seeds linearly — the key is Python-only — so --seeding-mode linear is the "
        "setting under which the two engines are actually comparable.",
    )
    ap.add_argument("--java", action="store_true", help="also run a single Java 4.4.1 cross-check")
    ap.add_argument("--out", default="docs/baltic_stability_certification_2026-07-01.md")
    args = ap.parse_args()

    params = _load_params(args.params)
    py_table = certify_python(params, args.years, tuple(args.seeds), args.seeding_mode)
    py_ok = _print_table("Python", py_table)

    lines = [
        "# Baltic stability — SP-A certification\n",
        f"**Params:** {args.params}  ·  **horizon:** {args.years} yr  ·  **seeds:** {args.seeds}"
        f"  ·  **seeding:** {args.seeding_mode or 'config default'}\n",
        "| species | persists | in-envelope | min biomass | final-decade mean range |",
        "|---|---|---|---|---|",
    ]
    for sp in FOCAL:
        t = py_table[sp]
        lines.append(
            f"| {sp} | {'✓' if t['persists'] else '✗'} | {'✓' if t['in_envelope'] else '✗'} "
            f"| {t['min_biomass']:.2e} | {t['late_mean_range']} |"
        )
    n_assessed, n_ind = len(ASSESSED), len(INDICATIVE)
    ind_ok = sum(1 for sp in INDICATIVE if py_table[sp]["persists"] and py_table[sp]["in_envelope"])
    verdict = (
        f"\n**Python verdict: {py_ok}/{n_assessed} ASSESSED species persistent & in-envelope.** "
        + (
            f"All {n_assessed} pass — candidate is certifiable; verify value round-trip before "
            "writing data/baltic."
            if py_ok == n_assessed
            else f"Not {n_assessed}/{n_assessed} — SP-B gate: the failing ASSESSED species are candidates "
            "params alone cannot stabilise; record whether sweeping their params moved them "
            "(structural vs tunable)."
        )
        + f"\n\n*Indicative tier: {ind_ok}/{n_ind} "
        f"({', '.join(f'{sp} w={TARGET_WEIGHT[sp]:g}' for sp in INDICATIVE)}).* "
        "These targets are **not ICES assessments** — ICES does not assess Baltic pikeperch, perch, "
        "smelt or stickleback. `biomass_targets.csv` sources them as literature estimates at "
        f"weight ≤ {INDICATIVE_MAX_WEIGHT}, noting the coarse grid under-resolves species "
        "concentrated in estuaries and lagoons. They are reported for information and are **not** "
        "part of the verdict; do not tune against them. "
        f"(Legacy all-species figure, for comparison with notes written before 2026-08-04: "
        f"{py_ok + ind_ok}/{len(FOCAL)}.)"
    )
    lines.append(verdict)

    if args.java:
        print("\n=== Java 4.4.1 cross-check (single run, Java's own RNG, staged via C2) ===")
        j_table = certify_java(params, args.years)
        if j_table:
            j_ok = _print_table("Java 4.4.1", j_table)
            py_surv = {sp for sp in FOCAL if py_table[sp]["persists"]}
            j_surv = {sp for sp in FOCAL if j_table[sp]["persists"]}
            agree = py_surv == j_surv
            lines.append(
                f"\n**Java cross-check: {j_ok}/{len(FOCAL)} persistent (single run, Java's own RNG "
                f"— Java 4.4.1 has no numeric seed).** Survivor sets "
                f"{'AGREE' if agree else 'DIFFER'} with Python — Python {sorted(py_surv)}, "
                f"Java {sorted(j_surv)}. Coarse consistency check only (Baltic is not bit-equal "
                "cross-engine); a DIFFER is a flag to inspect, not an automatic failure."
            )
        else:
            lines.append(
                "\n_Java cross-check unavailable (jar missing or run failed) — see console._"
            )

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote certification note to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
