"""Fishery-induced evolution (FIE) demonstration on Baltic cod.

Runs paired high-F vs low-F scenarios across multiple seeds, then plots
the mean cod imax trait trajectory with a multi-seed ribbon.

Usage: python scripts/run_fie_demo.py [--n-years 200] [--seeds 3] [--output-dir outputs/fie_demo]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path so that when
# this script is run from a git worktree the local osmose/ package takes
# precedence over any editable-install path pointing at the main repo.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in headless / CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.results import read_genetic_trait_means

MASTER = Path("data/baltic_ev/baltic_ev_all-parameters.csv")

# Baltic fishing uses the v4 fisheries-API (fisheries.enabled=true; per
# baltic_param-fishing.csv). When fisheries-API is active, `mortality.fishing.rate.*`
# legacy keys are ignored. Cod is targeted by fsh0 (fisheries.name.fsh0;trawlcod),
# so the per-fishery base rate is the correct override knob.
# Pin both fishing rate AND selectivity so the demo is self-documenting and
# does not silently inherit baltic's age-knife-edge selectivity (which would
# make the FIE selection differential on imax zero by design).
SCENARIOS = {
    "baltic_ev_high_f": {
        "fisheries.rate.base.fsh0": "0.6",
        "fisheries.selectivity.type.fsh0": "1",
        "fisheries.selectivity.l50.fsh0": "35.0",
        "fisheries.selectivity.slope.fsh0": "2.0",
    },
    "baltic_ev_low_f": {
        "fisheries.rate.base.fsh0": "0.1",
        "fisheries.selectivity.type.fsh0": "1",
        "fisheries.selectivity.l50.fsh0": "35.0",
        "fisheries.selectivity.slope.fsh0": "2.0",
    },
}

# Optional drift-only baseline. Enabled via --with-zero-f-control. Per
# caveat #6, F=0.1 still applies meaningful selection (Heino, Pauli &
# Dieckmann 2015); this arm gives a true neutral baseline at the cost of
# doubling wall-clock. Recommended for any published version of the demo.
ZERO_F_CONTROL = {
    "baltic_ev_zero_f": {
        "fisheries.rate.base.fsh0": "0.0",
        # Selectivity irrelevant when rate=0 but pin for reproducibility
        "fisheries.selectivity.type.fsh0": "1",
        "fisheries.selectivity.l50.fsh0": "35.0",
        "fisheries.selectivity.slope.fsh0": "2.0",
    },
}

_ALL_SCENARIOS = {**SCENARIOS, **ZERO_F_CONTROL}
_SCENARIO_COLORS = {
    "baltic_ev_high_f": "C3",
    "baltic_ev_low_f": "C0",
    "baltic_ev_zero_f": "C2",
}


def _build_cfg(scenario: str, n_years: int) -> dict[str, str]:
    cfg = OsmoseConfigReader().read(MASTER)
    if cfg.get("simulation.bioen.enabled", "false").lower() != "true":
        raise RuntimeError(
            "Ev-OSMOSE traits require simulation.bioen.enabled=true; "
            "baltic_ev is misconfigured."
        )
    cfg["simulation.time.nyear"] = str(n_years)
    cfg.update(_ALL_SCENARIOS[scenario])
    return cfg


def _run_one(scenario: str, seed: int, n_years: int, output_root: Path) -> None:
    out_dir = output_root / scenario / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    PythonEngine().run(_build_cfg(scenario, n_years), out_dir, seed=seed)


def _load(scenario: str, seeds: int, output_root: Path) -> pd.DataFrame:
    frames = []
    for s in range(seeds):
        ds = read_genetic_trait_means(output_root / scenario / f"seed{s}", prefix="osm")
        df = ds.sel(species_id=0, trait_name="imax")["mean"].to_dataframe().reset_index()
        df["seed"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-years", type=_positive_int, default=200)
    parser.add_argument("--seeds", type=_positive_int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fie_demo"))
    parser.add_argument(
        "--with-zero-f-control",
        action="store_true",
        help="Add a third F=0 arm as a drift-only neutral baseline. Doubles "
             "wall-clock but quantifies the low-F selection contribution "
             "from caveat #6.",
    )
    args = parser.parse_args()

    scenarios = dict(SCENARIOS)
    if args.with_zero_f_control:
        scenarios.update(ZERO_F_CONTROL)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        for s in range(args.seeds):
            print(f"Running {scenario} seed={s}...", flush=True)
            _run_one(scenario, s, args.n_years, args.output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario in scenarios:
        df = _load(scenario, args.seeds, args.output_dir)
        agg = df.groupby("Time")["mean"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["Time"], agg["mean"], color=_SCENARIO_COLORS[scenario], label=scenario)
        ax.fill_between(
            agg["Time"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=_SCENARIO_COLORS[scenario], alpha=0.2,
        )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Mean cod imax trait")
    ax.set_title("FIE on Baltic cod: mean imax trajectory under selective fishing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fie_imax_trajectory.png", dpi=150)

    # Print end-state summary
    for scenario in scenarios:
        df = _load(scenario, args.seeds, args.output_dir)
        end = df[df["Time"] == df["Time"].max()]["mean"]
        print(f"{scenario}: end-of-run mean imax = {end.mean():.4f} ± {end.std():.4f}")

    # Diagnostic: imax-binding fraction. If the cap (bioen_i_max) is rarely
    # binding because cod is prey-limited, the trait is a silent no-op
    # regardless of h². Reported per scenario.
    _print_imax_binding_diagnostic(args.output_dir, list(scenarios), args.seeds)


def _print_imax_binding_diagnostic(
    output_dir: Path, scenarios: list[str], seeds: int
) -> None:
    """Read ingestion vs imax-cap per cod-school-timestep and report what
    fraction of timesteps the cap was actually binding. < 30% means imax
    trait is structurally not the limiting constraint and the FIE signal
    will be drift-dominated."""
    print("\n=== imax-binding diagnostic ===")
    for scenario in scenarios:
        bind_fracs: list[float] = []
        for s in range(seeds):
            out_dir = output_dir / scenario / f"seed{s}"
            ingestion_csv = out_dir / "osm_bioen_ingestion_Simu0.csv"
            if not ingestion_csv.exists():
                print(f"{scenario} seed{s}: ingestion CSV missing; skipping")
                continue
            import pandas as pd
            df = pd.read_csv(ingestion_csv, sep=None, engine="python", comment="#")
            # Cap value for cod (sp0). Read from baltic_ev_param-bioen.csv.
            cap = 3.0  # matches Task 7.4's placeholder; if tuned, update.
            cod_col = "cod"
            if cod_col not in df.columns:
                print(f"{scenario} seed{s}: 'cod' column not in ingestion CSV; skipping")
                continue
            cod_series = df[cod_col]
            bind_frac = float((cod_series >= 0.95 * cap).mean())
            bind_fracs.append(bind_frac)
        if bind_fracs:
            import statistics
            mean_bind = statistics.mean(bind_fracs)
            print(f"{scenario}: cod imax-binding fraction across seeds = "
                  f"{mean_bind*100:.1f}% (per-seed: {[f'{x*100:.1f}%' for x in bind_fracs]})")
            if mean_bind < 0.30:
                print(f"  WARNING: imax-binding < 30% — FIE signal will be drift-dominated. "
                      f"imax trait is structurally not the limiting constraint in this config. "
                      f"Possible cause: declining cod growth potential in the eastern Baltic "
                      f"since the 1990s (Svedäng et al. 2024, "
                      f"https://doi.org/10.1002/ece3.70382, report L50 halving from 40 to 20cm "
                      f"attributed to deteriorating growth potential — NOTE the paper "
                      f"explicitly excludes simple prey-density / forage-fish mechanisms as "
                      f"the sole driver). FIE-direction test (Task 11) is unlikely to produce "
                      f"a meaningful result without first calibrating bioen params.")


if __name__ == "__main__":
    main()
