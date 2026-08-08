"""A/B: depletable LTL off vs on, under current production parameters (spec 2026-08-08 Phase 1).

Runs certify_python for each arm and reports per-species final-decade deltas plus the
identity-pinned gate verdict per arm. Measure first, certify second — this script issues NO
adoption verdict on its own; a human reads the report (spec: A/B before any certification
verdict; two-key decision rule in the Phase 1 plan, Task 3).

Arms: 'off' (depletion pinned false — explicit, so the arm stays meaningful after adoption),
'on' (the carried-over pre-split fitted rates), 'on-benthoslit' (REQUIRED: benthos at a
literature-plausible 0.03/step ~ P/B 0.7/yr, because the fitted 0.9116/step is 10-40x published
benthic turnover). --sensitivity adds a fourth arm with the a2_on_converged zoo rate.

    PYTHONPATH=. .venv/bin/python scripts/baltic_depletable_ab.py --out docs/baltic_depletable_ab_2026-08-08.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baltic_stability_certify import CERT_SEEDS, _print_table, certify_python  # noqa: E402

# CAUTION: fitted on the PRE-SPLIT 8-species config (phase1_results.json, 2026-07-10, before the
# cod E/W disaggregation); carried over as a plausible prior, NOT an optimum of this layout.
_FITTED_ZOO = "0.911553421016705"
BENTHOS_LIT_RATE = "0.03"  # per-step ~ P/B 0.7/yr; published benthic macrofauna P/B ~0.5-3/yr
A2_SENSITIVITY_ZOO_RATE = "1.0580953986747008"  # a2_on_converged (8-species co-fit; stale here)

DEPLETION_KEYS = {
    "ltl.depletable.enabled": "true",
    "ltl.depletable.floor": "0.05",
    "species.regrowth.rate.sp9": "5.0",  # Diatoms — phyto pinned near-reset (enable_a2_base_config)
    "species.regrowth.rate.sp10": "5.0",  # Dinoflagellates
    "species.regrowth.rate.sp11": _FITTED_ZOO,  # Microzooplankton
    "species.regrowth.rate.sp12": _FITTED_ZOO,  # Mesozooplankton
    "species.regrowth.rate.sp13": _FITTED_ZOO,  # Macrozooplankton
    "species.regrowth.rate.sp14": _FITTED_ZOO,  # Benthos — see BENTHOS_LIT_RATE caveat above
}

# Explicit, not {}: an empty override would mean 'repo default', which flips meaning the moment
# Task 4 commits depletion into data/baltic (post-adoption re-runs would measure on-vs-on).
ARM_OFF = {"ltl.depletable.enabled": "false"}

REQUIRED_PASS = ("cod_west", "cod_east", "herring", "sprat", "flounder", "perch", "stickleback")
TRACKED_ONLY = ("pikeperch", "smelt")


def identity_gate(table: dict) -> tuple[bool, list[str]]:
    """Identity-pinned gate: every REQUIRED_PASS species persists AND is in envelope."""
    failures = [
        sp for sp in REQUIRED_PASS if not (table[sp]["persists"] and table[sp]["in_envelope"])
    ]
    return (not failures, failures)


def _mid(row: dict) -> float:
    lo, hi = row["late_mean_range"]
    return (lo + hi) / 2.0


def make_report(tables: dict[str, dict], years: int, seeds: list[int]) -> str:
    arms = list(tables)
    base_name, others = arms[0], arms[1:]
    base = tables[base_name]
    mid_cols = " | ".join(f"{a} mid (t)" for a in arms)
    delta_cols = " | ".join(f"Δ {a} vs {base_name}" for a in others)
    lines = [
        "# Depletable LTL A/B (Phase 1, spec 2026-08-08)",
        "",
        f"**Arms:** {', '.join(arms)} · **horizon:** {years} yr · **seeds:** {list(seeds)}",
        "",
        f"| species | {mid_cols} | {delta_cols} | gated |",
        "|---|" + "---|" * (len(arms) + len(others) + 1),
    ]
    for sp in base:
        mids = {a: _mid(tables[a][sp]) for a in arms}
        cells = " | ".join(f"{mids[a]:,.0f}" for a in arms)
        deltas = " | ".join(
            f"{(mids[a] - mids[base_name]) / mids[base_name] * 100:+.1f}%"
            if mids[base_name]
            else "n/a"
            for a in others
        )
        gated = "yes" if sp in REQUIRED_PASS else "tracked only"
        lines.append(f"| {sp} | {cells} | {deltas} | {gated} |")
    for arm in arms:
        ok, failures = identity_gate(tables[arm])
        verdict = "PASS" if ok else f"FAIL ({', '.join(failures)})"
        lines.append("")
        lines.append(f"**GATE [{arm}]: {verdict}** (required: {', '.join(REQUIRED_PASS)})")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CERT_SEEDS))
    ap.add_argument(
        "--sensitivity",
        action="store_true",
        help="add a fourth arm with the a2_on_converged zoo rate (8-species co-fit)",
    )
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    arms: dict[str, dict[str, str]] = {
        "off": dict(ARM_OFF),
        "on": dict(DEPLETION_KEYS),
        "on-benthoslit": {**DEPLETION_KEYS, "species.regrowth.rate.sp14": BENTHOS_LIT_RATE},
    }
    if args.sensitivity:
        sens = dict(DEPLETION_KEYS)
        for sp in ("sp11", "sp12", "sp13", "sp14"):
            sens[f"species.regrowth.rate.{sp}"] = A2_SENSITIVITY_ZOO_RATE
        arms["on-a2conv"] = sens

    tables = {}
    for name, params in arms.items():
        print(f"\n=== arm: {name} ===")
        tables[name] = certify_python(params, args.years, args.seeds)
        _print_table(f"Python[{name}]", tables[name])

    report = make_report(tables, args.years, args.seeds)
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
