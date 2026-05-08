#!/usr/bin/env python3
"""Run the DSVM bioeconomic demo on top of the eec_full fixture.

Demonstrates the OSMOSE fleet-economics module (osmose.engine.economics)
end-to-end on an existing scenario by layering a 2-fleet DSVM config
on top of the eec_full base. Useful as:

1. **A copy-paste template** for users wanting to enable DSVM in their own
   scenario — the layered config keys are the complete activation surface.
2. **A runtime smoke test** that the fleet wiring (initialization →
   per-step decision → annual reset → output writing) works end-to-end on
   a real benchmark fixture, not just synthetic 1-cell test configs.
3. **A baseline-vs-DSVM comparison** for science-side validation: do
   vessels move toward catch-rich cells? Does revenue accrue? Does
   mortality on target species change vs the baseline?

The two demo fleets:

- **Demersal trawlers** — 5 vessels, target {cod (sp5), whiting (sp3),
  sole (sp7), plaice (sp8)}. Bottom-trawl gear, English-Channel-realistic
  prices (€/tonne).
- **Pelagic trawlers** — 5 vessels, target {herring (sp11), sardine
  (sp12), mackerel (sp10)}. Mid-water trawl gear, lower per-tonne prices
  but higher catch volumes.

Run from repo root:
    .venv/bin/python scripts/run_dsvm_demo.py [--years 3] [--seed 42] \\
        [--output-dir /tmp/dsvm-demo] [--baseline]

Outputs (under --output-dir):
    osm_biomass-Total-*.csv          species biomass time series
    econ_effort_<fleet>.csv          per-fleet effort by cell over time
    econ_revenue_<fleet>.csv         per-fleet vessel-level revenue
    econ_costs_<fleet>.csv           per-fleet vessel-level costs
    econ_profit_summary.csv          fleet-level profit summary
    dsvm_demo_summary.json           human-readable diagnostics

Pass --baseline to additionally run a no-DSVM comparison and write a
side-by-side species-biomass diff to dsvm_demo_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEC_CONFIG = PROJECT_ROOT / "data" / "eec_full" / "eec_all-parameters.csv"


# Species index → English Channel-realistic price (€ per tonne, 2020 reference)
# Demersal species fetch higher prices per tonne; pelagics fetch lower but
# catch volumes are larger. Sources: roughly aligned with EUMOFA market
# briefs and FAO commodity statistics (2018-2020 range), rounded for clarity.
_PRICES_EUR_PER_TONNE: dict[int, float] = {
    0: 800.0,    # lesserSpottedDogfish — low value, often by-catch
    1: 8000.0,   # redMullet — premium demersal
    2: 1500.0,   # pouting — mid value
    3: 1500.0,   # whiting — mid value
    4: 800.0,    # poorCod — low value
    5: 4000.0,   # cod — premium demersal
    6: 600.0,    # dragonet — low value
    7: 12000.0,  # sole — premium flatfish
    8: 6000.0,   # plaice — premium flatfish
    9: 1200.0,   # horseMackerel — pelagic, mid value
    10: 1500.0,  # mackerel — pelagic, mid-to-high value
    11: 800.0,   # herring — pelagic, lower per-tonne
    12: 1000.0,  # sardine — pelagic
    13: 5000.0,  # squids — high value
}

DEMERSAL_TARGETS = [5, 3, 7, 8]   # cod, whiting, sole, plaice
PELAGIC_TARGETS = [11, 12, 10]    # herring, sardine, mackerel


def _build_dsvm_overrides(
    raw: dict[str, str],
    *,
    enable_dsvm: bool,
) -> dict[str, str]:
    """Layer DSVM fleet config on top of an existing raw OSMOSE config."""
    if not enable_dsvm:
        # No-DSVM mode: return raw as-is. Used for the --baseline comparison.
        return raw

    overrides: dict[str, str] = dict(raw)
    overrides.update(
        {
            "simulation.economic.enabled": "true",
            "simulation.economic.rationality": "1.0",
            "simulation.economic.memory.decay": "0.7",
            "economic.fleet.number": "2",
        }
    )

    n_species = 14  # eec_full has 14 focal species

    # Fleet 0: Demersal trawlers
    overrides.update(_fleet_keys(
        fid=0,
        name="DemersalTrawlers",
        n_vessels=5,
        home_y=10,
        home_x=12,
        gear="bottom_trawl",
        max_days=200,
        fuel_cost=15.0,           # € per cell traversed
        operating_cost=2_000.0,   # € per vessel per day at sea
        target_species=DEMERSAL_TARGETS,
        n_species=n_species,
    ))

    # Fleet 1: Pelagic trawlers
    overrides.update(_fleet_keys(
        fid=1,
        name="PelagicTrawlers",
        n_vessels=5,
        home_y=5,
        home_x=8,
        gear="midwater_trawl",
        max_days=180,
        fuel_cost=20.0,
        operating_cost=2_500.0,
        target_species=PELAGIC_TARGETS,
        n_species=n_species,
    ))

    # Output enablement: spatial outputs are needed for fleet effort maps.
    overrides["output.spatial.enabled"] = "true"

    return overrides


def _fleet_keys(
    *,
    fid: int,
    name: str,
    n_vessels: int,
    home_y: int,
    home_x: int,
    gear: str,
    max_days: int,
    fuel_cost: float,
    operating_cost: float,
    target_species: list[int],
    n_species: int,
) -> dict[str, str]:
    """Generate the full set of `economic.fleet.*.fsh{fid}` keys for one fleet."""
    fsh = f"fsh{fid}"
    keys = {
        f"economic.fleet.name.{fsh}": name,
        f"economic.fleet.nvessels.{fsh}": str(n_vessels),
        f"economic.fleet.homeport.y.{fsh}": str(home_y),
        f"economic.fleet.homeport.x.{fsh}": str(home_x),
        f"economic.fleet.gear.{fsh}": gear,
        f"economic.fleet.max.days.{fsh}": str(max_days),
        f"economic.fleet.fuel.cost.{fsh}": str(fuel_cost),
        f"economic.fleet.operating.cost.{fsh}": str(operating_cost),
        f"economic.fleet.target.species.{fsh}": ",".join(str(s) for s in target_species),
    }
    for sp in range(n_species):
        keys[f"economic.fleet.price.sp{sp}.{fsh}"] = str(_PRICES_EUR_PER_TONNE.get(sp, 0.0))
        # Stock elasticity: how strongly fleet decisions react to local stock
        # density. Set to 1.0 for target species, 0.0 for non-target — vessels
        # avoid cells without target species but have neutral preference among
        # cells where all targets co-occur.
        elasticity = 1.0 if sp in target_species else 0.0
        keys[f"economic.fleet.stock.elasticity.sp{sp}.{fsh}"] = str(elasticity)
    return keys


def _run(
    raw_config: dict[str, str],
    output_dir: Path,
    seed: int,
) -> dict:
    """Run a single simulation and return summary metrics."""
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.output import write_outputs
    from osmose.engine.simulate import simulate

    cfg = EngineConfig.from_dict(raw_config)

    grid_file = raw_config.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            EEC_CONFIG.parent / grid_file,
            mask_var=raw_config.get("grid.var.mask", "mask"),
        )
    else:
        grid = Grid.from_dimensions(
            ny=int(raw_config.get("grid.nline", "1")),
            nx=int(raw_config.get("grid.ncolumn", "1")),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    outputs = simulate(cfg, grid, rng, output_dir=output_dir)
    elapsed = time.perf_counter() - t0
    write_outputs(outputs, output_dir, cfg)

    final = outputs[-1]
    biomass_by_species = {
        cfg.species_names[i] if i < len(cfg.species_names) else f"sp{i}": float(final.biomass[i])
        for i in range(len(final.biomass))
    }

    return {
        "elapsed_s": round(elapsed, 3),
        "n_steps": len(outputs),
        "final_biomass": biomass_by_species,
        "output_dir": str(output_dir),
    }


def _read_econ_outputs(output_dir: Path) -> dict:
    """Read economic CSVs that simulate() wrote — returns per-fleet summaries.

    The economics writer emits semicolon-delimited CSVs with a row per
    accumulation period (per-year) and a column per vessel — no header
    line. Use pd.read_csv with sep=';' and header=None to read.
    """
    import pandas as pd

    summary: dict = {"fleets": {}}
    profit_path = output_dir / "econ_profit_summary.csv"
    if profit_path.exists():
        df = pd.read_csv(profit_path, sep=";", header=None)
        summary["profit_summary"] = df.values.tolist()

    # Discover per-fleet files by glob — works even if fleet names change.
    for revenue_path in sorted(output_dir.glob("econ_revenue_*.csv")):
        fleet_name = revenue_path.stem.removeprefix("econ_revenue_")
        costs_path = output_dir / f"econ_costs_{fleet_name}.csv"
        effort_path = output_dir / f"econ_effort_{fleet_name}.csv"
        fleet_summary: dict = {}

        rev_df = pd.read_csv(revenue_path, sep=";", header=None)
        fleet_summary["total_revenue_eur"] = float(rev_df.values.sum())

        if costs_path.exists():
            costs_df = pd.read_csv(costs_path, sep=";", header=None)
            fleet_summary["total_costs_eur"] = float(costs_df.values.sum())
            fleet_summary["net_profit_eur"] = (
                fleet_summary["total_revenue_eur"] - fleet_summary["total_costs_eur"]
            )

        if effort_path.exists():
            fleet_summary["effort_csv"] = str(effort_path)
        summary["fleets"][fleet_name] = fleet_summary
    return summary


def _format_report(
    dsvm_run: dict,
    baseline_run: dict | None,
    econ_summary: dict,
) -> str:
    lines = [
        "=== DSVM bioeconomic demo ===",
        "",
        f"Run wall-time: {dsvm_run['elapsed_s']:.2f}s ({dsvm_run['n_steps']} steps)",
        f"Output dir:    {dsvm_run['output_dir']}",
        "",
        "--- Per-fleet economic summary ---",
    ]
    for fleet_name, fleet in econ_summary.get("fleets", {}).items():
        rev = fleet.get("total_revenue_eur", 0.0)
        costs = fleet.get("total_costs_eur", 0.0)
        profit = fleet.get("net_profit_eur", 0.0)
        lines.append(
            f"  {fleet_name:18s}  revenue €{rev:>14,.0f}  "
            f"costs €{costs:>14,.0f}  profit €{profit:>14,.0f}"
        )
    if not econ_summary.get("fleets"):
        lines.append("  (no economic outputs found — DSVM may not have been enabled)")
    lines += ["", "--- Final biomass (tonnes per species) ---"]
    for species, value in sorted(dsvm_run["final_biomass"].items()):
        if baseline_run is not None:
            base_val = baseline_run["final_biomass"].get(species, 0.0)
            ratio = value / base_val if base_val > 0 else float("nan")
            lines.append(f"  {species:24s}  DSVM {value:>15,.1f}  baseline {base_val:>15,.1f}  ×{ratio:.2f}")
        else:
            lines.append(f"  {species:24s}  {value:>15,.1f}")
    if baseline_run is not None:
        lines.append("")
        lines.append(
            f"Baseline run: {baseline_run['elapsed_s']:.2f}s, "
            f"output dir {baseline_run['output_dir']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--years", type=int, default=3, help="Years to simulate (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/dsvm-demo"),
        help="Output dir for DSVM run (default: /tmp/dsvm-demo)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also run a no-DSVM baseline for side-by-side comparison",
    )
    args = parser.parse_args(argv)

    if not EEC_CONFIG.exists():
        print(f"ERROR: missing fixture {EEC_CONFIG}", file=sys.stderr)
        return 2

    from osmose.config.reader import OsmoseConfigReader

    raw = OsmoseConfigReader().read(EEC_CONFIG)
    raw["simulation.time.nyear"] = str(args.years)

    dsvm_raw = _build_dsvm_overrides(raw, enable_dsvm=True)
    print(f"Running DSVM-enabled simulation ({args.years} yr) → {args.output_dir}")
    dsvm_run = _run(dsvm_raw, args.output_dir, args.seed)
    econ_summary = _read_econ_outputs(args.output_dir)

    baseline_run = None
    if args.baseline:
        baseline_dir = args.output_dir.parent / f"{args.output_dir.name}-baseline"
        print(f"Running baseline (no DSVM) → {baseline_dir}")
        baseline_raw = _build_dsvm_overrides(raw, enable_dsvm=False)
        baseline_run = _run(baseline_raw, baseline_dir, args.seed)

    report_text = _format_report(dsvm_run, baseline_run, econ_summary)
    print(report_text)

    summary_path = args.output_dir / "dsvm_demo_summary.json"
    summary_path.write_text(json.dumps({
        "dsvm_run": dsvm_run,
        "baseline_run": baseline_run,
        "econ_summary": econ_summary,
        "n_years": args.years,
        "seed": args.seed,
    }, indent=2))
    print(f"\nSummary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
